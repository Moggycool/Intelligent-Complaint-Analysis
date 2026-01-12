from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import gradio as gr

from src.rag_pipeline import RAGPipeline

APP_TITLE = "Intelligent Complaint Analysis (RAG)"
APP_SUBTITLE = "Ask questions about customer complaints and verify answers with sources."

_PIPELINE: Dict[str, Optional[RAGPipeline]] = {"instance": None}

# Gradio 6 "messages" format:
# [{"role": "user"|"assistant"|"system", "content": "..."}]
ChatMessages = List[Dict[str, str]]


class SourceChunk(dict):
    """{"rank": int, "score": float|None, "text": str, "meta": dict}"""


def _init_pipeline() -> RAGPipeline:
    if _PIPELINE["instance"] is None:
        try:
            _PIPELINE["instance"] = RAGPipeline(
                retriever=None, generator=None)  # type: ignore
        except TypeError:
            _PIPELINE["instance"] = RAGPipeline(None, None)  # type: ignore
    return _PIPELINE["instance"]


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
                if meta.get(k):
                    meta_bits.append(f"{k}: {meta.get(k)}")

        meta_str = (" | " + " • ".join(meta_bits)) if meta_bits else ""
        score_str = f" (score: {score:.4f})" if isinstance(
            score, (int, float)) else ""
        lines.append(f"### Source {rank}{score_str}{meta_str}\n{text}")

    return "\n\n".join(lines).strip()


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
    # RAGResult-like object
    if hasattr(result, "answer") and hasattr(result, "sources"):
        answer = str(getattr(result, "answer", "") or "").strip()
        raw_sources = getattr(result, "sources", []) or []
        return answer, _normalize_raw_sources(raw_sources)

    # dataclass -> dict
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(cast(Any, result))

    # dict
    if isinstance(result, dict):
        answer = result.get("answer") or result.get(
            "output") or result.get("response") or ""
        raw_sources = (
            result.get("sources")
            or result.get("contexts")
            or result.get("chunks")
            or result.get("retrieved_chunks")
            or []
        )
        return str(answer).strip(), _normalize_raw_sources(raw_sources)

    # tuple/list (answer, sources)
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return str(result[0]).strip(), _normalize_raw_sources(result[1])

    return str(result).strip(), []


def _call_pipeline(pipe: Any, question: str) -> Any:
    for name in ("run", "answer", "ask", "generate", "predict"):
        if hasattr(pipe, name):
            fn = getattr(pipe, name)
            if callable(fn):
                return fn(question)
    if callable(pipe):
        return pipe(question)
    raise TypeError(
        "Pipeline has no callable method (run/answer/ask/generate/predict or __call__).")


def _append_turn(history: ChatMessages, user_text: str, assistant_text: str) -> ChatMessages:
    history = history or []
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return history


def ask(question: str, history: ChatMessages) -> Tuple[ChatMessages, str, ChatMessages]:
    question = (question or "").strip()
    history = history or []
    if not question:
        return history, "", history

    pipe = _init_pipeline()
    try:
        result = _call_pipeline(pipe, question)
        answer, sources = _extract_answer_and_sources(result)
    except Exception as e:
        answer = f"Error while running the pipeline: {type(e).__name__}: {e}"
        sources = []

    new_history = _append_turn(history, question, answer)
    return new_history, _format_sources(sources), new_history


def ask_stream(question: str, history: ChatMessages):
    question = (question or "").strip()
    history = history or []
    if not question:
        yield history, "", history
        return

    pipe = _init_pipeline()

    # Try native streaming if provided by your pipeline
    if hasattr(pipe, "stream") and callable(getattr(pipe, "stream")):
        partial = ""
        # Add the user message once, then keep updating the assistant message
        base = history + [{"role": "user", "content": question}]
        try:
            for token in pipe.stream(question):  # type: ignore[attr-defined]
                partial += str(token)
                temp = base + [{"role": "assistant", "content": partial}]
                yield temp, "", temp
        except Exception as e:
            err = f"Error while streaming: {type(e).__name__}: {e}"
            new_history = base + [{"role": "assistant", "content": err}]
            yield new_history, "", new_history
            return

        final_history = base + [{"role": "assistant", "content": partial}]
        yield final_history, "", final_history
        return

    # Fallback: run once then typewriter
    try:
        result = _call_pipeline(pipe, question)
        answer, sources = _extract_answer_and_sources(result)
    except Exception as e:
        answer = f"Error while running the pipeline: {type(e).__name__}: {e}"
        sources = []

    base = history + [{"role": "user", "content": question}]
    typed = ""
    for ch in answer:
        typed += ch
        temp = base + [{"role": "assistant", "content": typed}]
        yield temp, "", temp
        time.sleep(0.005)

    final_history = base + [{"role": "assistant", "content": answer}]
    yield final_history, _format_sources(sources), final_history


def clear_all() -> Tuple[ChatMessages, str, str, ChatMessages]:
    return [], "Sources will appear here after you ask a question.", "", []


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_SUBTITLE}")

    with gr.Row():
        with gr.Column(scale=3):
            # Gradio 6: use messages mode
            chatbot = gr.Chatbot(label="Conversation", height=420)
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Type your question about the complaints…",
                lines=2,
            )
            with gr.Row():
                ask_btn: Any = gr.Button("Ask", variant="primary")
                clear_btn: Any = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## Sources (Retrieved Evidence)")
            sources_box = gr.Markdown(
                value="Sources will appear here after you ask a question.")

    state = gr.State([])  # stores ChatMessages

    ask_evt = getattr(ask_btn, "click")(
        fn=ask_stream,  # change to ask for non-streaming
        inputs=[question_box, state],
        outputs=[chatbot, sources_box, state],
    )
    ask_evt.then(fn=lambda: "", inputs=None, outputs=question_box)

    getattr(clear_btn, "click")(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, sources_box, question_box, state],
    )

if __name__ == "__main__":
    demo.launch()
