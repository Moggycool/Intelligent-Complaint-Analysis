"""Prompt engineering utilities for RAG-based complaint analysis."""
from typing import List


def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Build a grounded prompt for the LLM.

    Parameters
    ----------
    context_chunks : List[str]
        Retrieved complaint excerpts.
    question : str
        User question.

    Returns
    -------
    str
        Formatted prompt.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.

Use ONLY the information provided in the context below.
If the context does not contain enough information to answer the question,
clearly state that you do not have sufficient information.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()
