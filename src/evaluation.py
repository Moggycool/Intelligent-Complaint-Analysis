"""Qualitative evaluation utilities for RAG outputs."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import pandas as pd


def _parse_yes_no_schema(answer: str) -> Tuple[str, str]:
    """
    Parse the strict two-line schema:
      MENTIONED: Yes / No / I don't know
      EVIDENCE: ...

    Returns (mentioned, evidence). If not parseable, returns ("", "").
    """
    if not answer or not isinstance(answer, str):
        return "", ""

    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "", ""

    m_line, e_line = lines[0], lines[1]

    if not m_line.lower().startswith("mentioned:"):
        return "", ""
    if not e_line.lower().startswith("evidence:"):
        return "", ""

    mentioned = m_line.split(":", 1)[1].strip()
    evidence = e_line.split(":", 1)[1].strip()
    return mentioned, evidence


def evaluate_responses(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a qualitative evaluation table (Task 3 spec).
    Adds parsed Yes/No fields when present.
    """
    rows = []

    for r in results:
        sources = (r.get("sources") or [])[:2]
        answer = r.get("answer", "")

        mentioned, evidence = _parse_yes_no_schema(answer)

        def fmt_source(s: Dict[str, Any]) -> str:
            if not isinstance(s, dict):
                return str(s)

            # Retriever populates `text`; keep compatibility with other possible keys too
            raw_text = (s.get("text") or s.get(
                "chunk_text") or s.get("content") or "")
            snippet = str(raw_text).replace("\n", " ").strip()
            snippet = (snippet[:220] +
                       "...") if len(snippet) > 220 else snippet

            score_val = s.get("score")
            if isinstance(score_val, (int, float)):
                score_str = f"{float(score_val):.4f}"
            else:
                score_str = "NA"

            # doc_id may not exist in your metadata; index_id always exists from Retriever
            doc_id = s.get("doc_id")
            if doc_id is None:
                doc_id = s.get("index_id")

            return (
                f"complaint_id={s.get('complaint_id')} | product={s.get('product')} | "
                f"doc_id={doc_id} | score={score_str} :: {snippet}"
            )

        rows.append(
            {
                "Question": r.get("question", ""),
                "Generated Answer": answer,
                "MENTIONED (parsed)": mentioned,   # blank if not in schema
                "EVIDENCE (parsed)": evidence,     # blank if not in schema
                "Retrieved Sources (show 1-2)": " || ".join(fmt_source(s) for s in sources),
                "Quality Score (1-5)": "TBD",
                "Comments/Analysis": "TBD (manual review)",
            }
        )

    return pd.DataFrame(rows)
