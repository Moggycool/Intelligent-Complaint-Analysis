"""Qualitative evaluation utilities for RAG outputs."""
from typing import List, Dict
import pandas as pd


def evaluate_responses(results: List[Dict]) -> pd.DataFrame:
    """
    Create a qualitative evaluation table.

    Parameters
    ----------
    results : List[Dict]
        RAG outputs containing question, answer, and sources.

    Returns
    -------
    pd.DataFrame
        Evaluation table for reporting.
    """
    rows = []

    for r in results:
        rows.append({
            "Question": r["question"],
            "Generated Answer": r["answer"],
            "Retrieved Sources": " | ".join(r["sources"]),
            "Quality Score (1–5)": "",
            "Comments / Analysis": ""
        })

    return pd.DataFrame(rows)
