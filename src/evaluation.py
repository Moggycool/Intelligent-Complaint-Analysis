"""Qualitative evaluation utilities for RAG outputs."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import pandas as pd


def _parse_yes_no_schema(answer: str) -> Tuple[str, str]:
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


def _parse_issues_schema(answer: str) -> Tuple[str, int]:
    """
    Parse aggregate schema:
      ISSUES:
      - ...
      - ...
    Returns (issues_joined, num_bullets). If not parseable, ("", 0).
    """
    if not answer or not isinstance(answer, str):
        return "", 0

    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if not lines or lines[0] != "ISSUES:":
        return "", 0

    bullets = [ln for ln in lines[1:] if ln.startswith(("-", "*", "•"))]
    return " | ".join(bullets), len(bullets)


def evaluate_responses(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for r in results:
        sources = (r.get("sources") or [])[:2]
        answer = r.get("answer", "")

        mentioned, evidence = _parse_yes_no_schema(answer)
        issues, num_bullets = _parse_issues_schema(answer)

        def fmt_source(s: Dict[str, Any]) -> str:
            if not isinstance(s, dict):
                return str(s)

            raw_text = (s.get("text") or s.get(
                "chunk_text") or s.get("content") or "")
            snippet = str(raw_text).replace("\n", " ").strip()
            snippet = (snippet[:220] +
                       "...") if len(snippet) > 220 else snippet

            score_val = s.get("score")
            score_str = f"{float(score_val):.4f}" if isinstance(
                score_val, (int, float)) else "NA"

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

                # Yes/No parse fields (blank when not applicable)
                "MENTIONED (parsed)": mentioned,
                "EVIDENCE (parsed)": evidence,

                # Aggregate parse fields (blank when not applicable)
                "ISSUES (parsed)": issues,
                "NUM_BULLETS (parsed)": num_bullets,

                "Retrieved Sources (show 1-2)": " || ".join(fmt_source(s) for s in sources),
                "Quality Score (1-5)": "TBD",
                "Comments/Analysis": "TBD (manual review)",
            }
        )

    return pd.DataFrame(rows)
