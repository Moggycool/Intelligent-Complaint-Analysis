from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import gradio as gr

from src.rag_pipeline import RAGPipeline, RAGResult

# These exist in your tree; we will use them via best-effort factories.
import src.vector_store as vector_store_mod
import src.retriever as retriever_mod
import src.generator as generator_mod


APP_TITLE = "Intelligent Complaint Analysis (RAG)"
APP_SUBTITLE = "Ask questions about customer complaints and verify answers with sources."

# Expected local vector store directory
VSTORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
FAISS_INDEX_PATH = os.path.join(VSTORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VSTORE_DIR, "metadata.pkl")

# Older Gradio Chatbot expects: List[Tuple[user, assistant]]
ChatHistory = List[Tuple[str, str]]

_PIPELINE: Dict[str, Optional[RAGPipeline]] = {"instance": None}


class SourceChunk(dict):
    """{"rank": int, "score": float|None, "text": str, "meta": dict}"""


# ----------------------------
# Helpers: dynamic builders (robust across small API differences)
# ----------------------------
def _build_vector_store() -> Any:
    """
    Loads your FAISS vector store from ./vector_store.

    This function tries common patterns:
      - VectorStore.load(...)
      - load_vector_store(...)
      - load_faiss_index(...)
      - VectorStore(index_path=..., metadata_path=...)
    """
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        raise RuntimeError(
            "Vector store files not found.\n"
            f"Expected:\n- {FAISS_INDEX_PATH}\n- {METADATA_PATH}\n\n"
            "Run Task 2/3 ingestion & indexing first (or fix the output paths)."
        )

    # 1) module-level loader functions
    for fn_name in ("load", "load_vector_store", "load_faiss_index", "load_index"):
        if hasattr(vector_store_mod, fn_name) and callable(getattr(vector_store_mod, fn_name)):
            fn = getattr(vector_store_mod, fn_name)
            try:
                return fn(VSTORE_DIR)
            except TypeError:
                # some loaders want explicit paths
                return fn(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)

    # 2) class-based loader
    for cls_name in ("VectorStore", "FaissVectorStore", "FAISSVectorStore"):
        if hasattr(vector_store_mod, cls_name):
            cls = getattr(vector_store_mod, cls_name)
            # try common constructors / classmethods
            if hasattr(cls, "load") and callable(getattr(cls, "load")):
                try:
                    return cls.load(VSTORE_DIR)
                except TypeError:
                    return cls.load(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
            try:
                return cls(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
            except TypeError:
                return cls(FAISS_INDEX_PATH, METADATA_PATH)

    raise RuntimeError(
        "Could not construct vector store from src/vector_store.py. "
        "Please share src/vector_store.py so I can align the exact API."
    )


def _build_retriever(vstore: Any) -> Any:
    """
    Builds a Retriever object that has .retrieve(query, k=...).
    """
    # If the vector store itself offers as_retriever()
    if hasattr(vstore, "as_retriever") and callable(getattr(vstore, "as_retriever")):
        return vstore.as_retriever()

    # Retriever class/function in src/retriever.py
    for cls_name in ("Retriever", "VectorStoreRetriever", "FaissRetriever", "FAISSRetriever"):
        if hasattr(retriever_mod, cls_name):
            cls = getattr(retriever_mod, cls_name)
            try:
                return cls(vstore)
            except TypeError:
                # sometimes needs keyword
                return cls(vector_store=vstore)

    # module-level factory
    for fn_name in ("build_retriever", "get_retriever", "make_retriever"):
        if hasattr(retriever_mod, fn_name) and callable(getattr(retriever_mod, fn_name)):
            fn = getattr(retriever_mod, fn_name)
            return fn(vstore)

    raise RuntimeError(
        "Could not construct retriever from src/retriever.py. "
        "Please share src/retriever.py so I can align the exact API."
    )


def _build_generator() -> Any:
    """
    Builds your generator (LLM wrapper) from src/generator.py.
    Must expose .generate(prompt, generate_kwargs=...) returning an object with .text
    (per your rag_pipeline.py).
    """
    # Common patterns
    for fn_name in ("build_generator", "get_generator", "load_generator", "make_generator"):
        if hasattr(generator_mod, fn_name) and callable(getattr(generator_mod, fn_name)):
            return getattr(generator_mod, fn_name)()

    for cls_name in ("Generator", "TextGenerator", "LLMGenerator", "HFGenerator"):
        if hasattr(generator_mod, cls_name):
            cls = getattr(generator_mod, cls_name)
            try:
                return cls()
            except TypeError:
                # if it needs args, we need the file to know what
                break

    raise RuntimeError(
        "Could not construct generator from src/generator.py. "
        "Please share src/generator.py so I can align the exact API/constructor args."
    )


def _init_pipeline() -> RAGPipeline:
    """
    Proper pipeline init: loads vector store -> retriever, builds generator -> pipeline.
    """
    if _PIPELINE["instance"] is None:
        vstore = _build_vector_store()
        retriever = _build_retriever(vstore)
        generator = _build_generator()

        pipe = RAGPipeline(retriever=retriever, generator=generator, k=3)

        # Defensive checks (clear error instead of NoneType crash)
        if getattr(pipe, "retriever", None) is None:
            raise RuntimeError(
                "Pipeline retriever is None (failed to initialize).")
        if getattr(pipe, "generator", None) is None:
            raise RuntimeError(
                "Pipeline generator is None (failed to initialize).")

        _PIPELINE["instance"] = pipe

    return _PIPELINE["instance"]


# ----------------------------
# Result unwrapping + sources
# ----------------------------
def _normalize_raw_sources(raw_sources: Any) -> List[SourceChunk]:
    if not raw_sources:
        return []
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]

    out: List[SourceChunk] = []
    for i, item in enumerate(raw_sources, start=1):
        if isinstance(item, str):
            out.append(SourceChunk(rank=i, score=None, text=item, meta={}))
            continue
        if isinstance(item, dict):
            text = item.get("text") or item.get(
                "chunk") or item.get("content") or ""
            score = item.get("score")
            meta = item.get("metadata") or item.get("meta")
            if not isinstance(meta, dict):
                meta = {k: v for k, v in item.items() if k not in (
                    "text", "chunk", "content", "score")}
            out.append(SourceChunk(rank=i, score=score,
                       text=text, meta=meta or {}))
            continue
        out.append(SourceChunk(rank=i, score=None, text=str(item), meta={}))
    return out


def _extract_answer_and_sources(result: Any) -> Tuple[str, List[SourceChunk]]:
    # Your RAGResult
    if isinstance(result, RAGResult) or (hasattr(result, "answer") and hasattr(result, "sources")):
        answer = str(getattr(result, "answer", "") or "").strip()
        raw_sources = getattr(result, "sources", []) or []
        return answer, _normalize_raw_sources(raw_sources)

    # dataclass instance -> dict (Pylance-safe)
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(cast(Any, result))

    if isinstance(result, dict):
        answer = result.get("answer") or result.get(
            "output") or result.get("response") or ""
        raw_sources = result.get("sources") or result.get(
            "contexts") or result.get("chunks") or []
        return str(answer).strip(), _normalize_raw_sources(raw_sources)

    if isinstance(result, (tuple, list)) and len(result) == 2:
        return str(result[0]).strip(), _normalize_raw_sources(result[1])

    return str(result).strip(), []


def _format_sources(chunks: List[SourceChunk]) -> str:
    if not chunks:
        return "No sources were retrieved for this question."

    lines: List[str] = []
    for ch in chunks:
        rank = ch.get("rank", 0)
        score = ch.get("score", None)
        text = (ch.get("text", "") or "").strip()
        meta = ch.get("meta", {}) or {}

        meta_bits: List[str] = []
        if isinstance(meta, dict):
            for k in ["complaint_id", "product", "issue", "company", "date", "doc_id"]:
                if meta.get(k) is not None and meta.get(k) != "":
                    meta_bits.append(f"{k}: {meta.get(k)}")

        meta_str = (" | " + " • ".join(meta_bits)) if meta_bits else ""
        score_str = f" (score: {score:.4f})" if isinstance(
            score, (int, float)) else ""

        lines.append(f"### Source {rank}{score_str}{meta_str}\n{text}")

    return "\n\n".join(lines).strip()


# ----------------------------
# App actions (Gradio)
# ----------------------------
def ask_stream(question: str, history: ChatHistory):
    question = (question or "").strip()
    history = history or []

    if not question:
        yield history, "", history
        return

    try:
        pipe = _init_pipeline()
        # IMPORTANT: your pipeline uses .answer()
        result = pipe.answer(question)
        answer, sources = _extract_answer_and_sources(result)
    except Exception as e:
        answer = f"Error while running the pipeline: {type(e).__name__}: {e}"
        sources = []

    typed = ""
    for ch in answer:
        typed += ch
        temp_history = history + [(question, typed)]
        yield temp_history, "", temp_history
        time.sleep(0.003)

    new_history = history + [(question, answer)]
    yield new_history, _format_sources(sources), new_history


def clear_all() -> Tuple[ChatHistory, str, str, ChatHistory]:
    return [], "Sources will appear here after you ask a question.", "", []


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_SUBTITLE}")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=420)
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Type your question about the complaints…",
                lines=2,
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## Sources (Retrieved Evidence)")
            sources_box = gr.Markdown(
                value="Sources will appear here after you ask a question.")

    state = gr.State([])  # ChatHistory

    ask_evt = ask_btn.click(  # pylint: disable=no-member
        fn=ask_stream,
        inputs=[question_box, state],
        outputs=[chatbot, sources_box, state],
    )
    ask_evt.then(  # pylint: disable=no-member
        fn=lambda: "",
        inputs=None,
        outputs=question_box,
    )

    clear_btn.click(  # pylint: disable=no-member
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, sources_box, question_box, state],
    )

if __name__ == "__main__":
    demo.launch()
