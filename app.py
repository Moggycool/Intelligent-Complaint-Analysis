"""
Task 4: Interactive Chat Interface (Gradio)
- Question input box
- Ask button
- Answer display
- Sources display (retrieved chunks used)
- Clear button
- Optional streaming
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---- Import your pipeline (adjust these imports to match your repo) ----
# Common possibilities:
#   from rag_pipeline import RAGPipeline
from src.rag_pipeline import RAGPipeline
#


# ---------------------------- Config ----------------------------

APP_TITLE = "Intelligent Complaint Analysis (RAG)"
APP_SUBTITLE = "Ask questions about customer complaints and verify answers with sources."

# If your pipeline needs paths/config, set them here.
# Keep it simple for non-technical users.
_PIPELINE: Dict[str, Optional[RAGPipeline]] = {"instance": None}


@dataclass
class SourceChunk:
    """ A retrieved source chunk with metadata. """
    rank: int
    score: Optional[float]
    text: str
    meta: Dict[str, Any]


def _init_pipeline() -> RAGPipeline:
    """
    Initialize the RAG pipeline once.
    Adjust this if your RAGPipeline constructor requires arguments.
    """
    if _PIPELINE["instance"] is None:
        # Try common constructor signatures to be robust across implementations.
        try:
            _PIPELINE["instance"] = RAGPipeline()
        except TypeError:
            try:
                # Some implementations expect named parameters
                _PIPELINE["instance"] = RAGPipeline(
                    retriever=None, generator=None)  # type: ignore
            except TypeError:
                # Fallback to positional Nones if required
                _PIPELINE["instance"] = RAGPipeline(None, None)  # type: ignore
    return _PIPELINE["instance"]


def _format_sources(chunks: List[SourceChunk]) -> str:
    """Readable sources block (non-technical, but transparent)."""
    if not chunks:
        return "No sources were retrieved for this question."

    lines = []
    for ch in chunks:
        meta_bits = []
        if ch.meta:
            # show only helpful, human-friendly metadata if present
            for k in ["complaint_id", "product", "issue", "company", "date"]:
                if k in ch.meta and ch.meta[k]:
                    meta_bits.append(f"{k}: {ch.meta[k]}")
        meta_str = (" | " + " • ".join(meta_bits)) if meta_bits else ""
        score_str = f" (score: {ch.score:.4f})" if isinstance(
            ch.score, (int, float)) else ""

        lines.append(
            f"### Source {ch.rank}{score_str}{meta_str}\n"
            f"{ch.text.strip()}"
        )

    return "\n\n".join(lines).strip()


def _extract_sources_from_pipeline_result(result: Any) -> Tuple[str, List[SourceChunk]]:
    """
    Tries to normalize different possible return shapes from your pipeline.

    We aim for:
      answer: str
      sources: list of SourceChunk(text, score, meta)
    """
    # Case A: pipeline returns dict: {"answer": "...", "contexts": [...]} etc.
    if isinstance(result, dict):
        answer = (
            result.get("answer")
            or result.get("output")
            or result.get("response")
            or ""
        )

        raw_sources = (
            result.get("sources")
            or result.get("contexts")
            or result.get("chunks")
            or result.get("retrieved_chunks")
            or []
        )

        sources: List[SourceChunk] = []
        for i, item in enumerate(raw_sources, start=1):
            # item might be a string chunk
            if isinstance(item, str):
                sources.append(SourceChunk(
                    rank=i, score=None, text=item, meta={}))
                continue

            # item might be dict with text/score/metadata
            if isinstance(item, dict):
                text = item.get("text") or item.get(
                    "chunk") or item.get("content") or ""
                score = item.get("score")
                meta = item.get("metadata") or item.get("meta") or {}
                sources.append(SourceChunk(
                    rank=i, score=score, text=text, meta=meta))
                continue

            # fallback
            sources.append(SourceChunk(
                rank=i, score=None, text=str(item), meta={}))

        return str(answer).strip(), sources

    # Case B: pipeline returns tuple: (answer, sources)
    if isinstance(result, (tuple, list)) and len(result) == 2:
        answer = str(result[0]).strip()
        raw_sources = result[1] or []
        sources: List[SourceChunk] = []
        for i, item in enumerate(raw_sources, start=1):
            if isinstance(item, str):
                sources.append(SourceChunk(
                    rank=i, score=None, text=item, meta={}))
            elif isinstance(item, dict):
                text = item.get("text") or item.get(
                    "chunk") or item.get("content") or ""
                score = item.get("score")
                meta = item.get("metadata") or item.get("meta") or {}
                sources.append(SourceChunk(
                    rank=i, score=score, text=text, meta=meta))
            else:
                sources.append(SourceChunk(
                    rank=i, score=None, text=str(item), meta={}))
        return answer, sources

    # Case C: pipeline returns plain string
    return str(result).strip(), []


# ---------------------------- Core App Logic ----------------------------

def ask(question: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """
    Non-streaming ask: updates chat history and returns sources markdown.
    """
    question = (question or "").strip()
    if not question:
        return chat_history, ""

    pipe = _init_pipeline()

    # IMPORTANT: adjust this call to match your pipeline API.
    # Common patterns:
    #   result = pipe.run(question)
    #   result = pipe.answer(question)
    #   result = pipe(question)
    #
    # Use a helper to robustly call the pipeline without assuming it's callable.
    result = _call_pipeline(pipe, question)

    answer, sources = _extract_sources_from_pipeline_result(result)
    chat_history = chat_history + [(question, answer)]
    sources_md = _format_sources(sources)
    return chat_history, sources_md


def _call_pipeline(pipe, question):
    """Call the pipeline using common method names without assuming the object is callable."""
    for name in ("run", "answer", "ask", "generate", "predict"):
        if hasattr(pipe, name):
            attr = getattr(pipe, name)
            if callable(attr):
                return attr(question)
    # fallback to __call__ if object itself is callable
    if callable(pipe):
        return pipe(question)
    raise TypeError(
        "Pipeline object has no callable method (expected run/answer/ask/generate/predict or __call__)"
    )


def ask_stream(question: str, chat_history: List[Tuple[str, str]]):
    """
    Streaming: yields intermediate chat updates + sources at the end.
    - If your generator/pipeline supports streaming tokens, plug it in here.
    - Otherwise we do a simple "typewriter" stream for UX.
    """
    question = (question or "").strip()
    if not question:
        yield chat_history, ""
        return

    pipe = _init_pipeline()

    # --- Real streaming hook (optional) ---
    # If your pipeline has something like pipe.stream(question) that yields tokens:
    #   for token in pipe.stream(question): ...
    #
    # Otherwise: run once and typewriter the final answer.
    if hasattr(pipe, "stream"):
        partial = ""
        for token in pipe.stream(question):  # type: ignore
            partial += str(token)
            # show partial answer in last assistant bubble
            temp_history = chat_history + [(question, partial)]
            yield temp_history, ""  # sources shown at end
        # after streaming completes, attempt to fetch sources if available
        # (some streaming APIs provide sources separately; adapt if you have it)
        yield chat_history + [(question, partial)], ""
        return

    # Fallback: non-streaming run, then typewriter effect
    result = _call_pipeline(pipe, question)

    answer, sources = _extract_sources_from_pipeline_result(result)

    typed = ""
    for ch in answer:
        typed += ch
        yield chat_history + [(question, typed)], ""
        time.sleep(0.005)  # small delay for “token-like” feel

    sources_md = _format_sources(sources)
    yield chat_history + [(question, answer)], sources_md


def clear_all():
    """Resets conversation + sources."""
    return [], "", ""


# ---------------------------- UI ----------------------------

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_SUBTITLE}")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                show_copy_button=True,
            )
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
                value="Sources will appear here after you ask a question."
            )

    # hidden state: chat history
    state = gr.State([])  # List[Tuple[user, assistant]]

    # Wire buttons
    # Streaming recommended: button triggers generator that yields updates
    ask_btn.click(
        fn=ask_stream,  # change to ask (non-streaming) if you prefer
        inputs=[question_box, state],
        outputs=[chatbot, sources_box],
    ).then(
        fn=lambda: "",  # clear input box after submit
        inputs=None,
        outputs=question_box,
    )

    # Keep internal state synced from chatbot
    chatbot.change(fn=lambda x: x, inputs=chatbot, outputs=state)

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, sources_box, question_box],
    ).then(
        fn=lambda: [],
        inputs=None,
        outputs=state,
    )

if __name__ == "__main__":
    demo.launch()
