"""
Prompt engineering utilities for RAG-based complaint analysis.
"""
from __future__ import annotations

from typing import List

REFUSAL = "I don't have enough information in the provided complaints to answer that."


def _clean_chunks(context_chunks: List[str]) -> List[str]:
    cleaned = [(c or "").strip() for c in context_chunks]
    return [c for c in cleaned if c]


def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Grounded free-form answer prompt.
    """
    cleaned_chunks = _clean_chunks(context_chunks)

    if not cleaned_chunks:
        return f"""You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Rules:
- Use ONLY the information in the provided context excerpts.
- If the context does not contain enough information to answer, say exactly:
  "{REFUSAL}"
- Do not guess or use outside knowledge.
- Keep the answer concise, factual, and grounded in the excerpts.

Context excerpts:
(none)

Question: {question}

Answer:
{REFUSAL}
""".strip()

    context = "\n\n".join(cleaned_chunks)

    return f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Rules:
- Use ONLY the information in the provided context excerpts.
- If the context does not contain enough information to answer, say exactly:
  "{REFUSAL}"
- Do not guess or use outside knowledge.
- Keep the answer concise, factual, and grounded in the excerpts.

Context excerpts:
{context}

Question: {question}

Answer:
""".strip()


def build_yes_no_prompt(context_chunks: List[str], question: str) -> str:
    """
    Strict Yes/No/I don't know prompt with evidence quoting.

    Output MUST be exactly two lines:
      MENTIONED: Yes / No / I don't know
      EVIDENCE: <short quote from context OR REFUSAL>
    """
    cleaned_chunks = _clean_chunks(context_chunks)
    context = "\n".join(cleaned_chunks) if cleaned_chunks else "(none)"

    return f"""
You are a financial analyst assistant for CrediTrust.

Rules:
- Use ONLY the information in the context excerpts.
- Do not guess or use outside knowledge.
- Output EXACTLY two lines in the required format.
- EVIDENCE must be copied from the context (a short quote, ~5–20 words).
- If the context does not support an answer, respond with:
  MENTIONED: I don't know
  EVIDENCE: {REFUSAL}

Required format (exactly two lines):
MENTIONED: Yes / No / I don't know
EVIDENCE: <short quote from excerpts OR REFUSAL>

Context excerpts:
{context}

Question: {question}
""".strip()
