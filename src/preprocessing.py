"""Preprocessing functions for CFPB complaints dataset."""
import re
from typing import List
import pandas as pd


def filter_products_and_narratives(
    df: pd.DataFrame,
    product_col: str,
    text_col: str,
    allowed_products: List[str]
) -> pd.DataFrame:
    """
    Filter dataset to required products and non-empty narratives.
    """
    filtered = df[
        df[product_col].isin(allowed_products) &
        df[text_col].notna() &
        (df[text_col].str.strip() != "")
    ].copy()

    return filtered


def clean_narrative_text(text: str) -> str:
    """
    Clean complaint narrative text for embedding.
    """
    text = text.lower()

    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"this complaint is regarding",
        r"consumer complaint narrative"
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_text_cleaning(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Apply text cleaning to narratives.
    """
    df["cleaned_narrative"] = df[text_col].apply(clean_narrative_text)
    return df


def get_narrative_length_stats(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Get statistics of narrative lengths in the dataset.
    """
    lengths = df[text_col].astype(str).apply(lambda x: len(x.split()))
    return lengths.describe()


def count_narrative_presence(df: pd.DataFrame, text_col: str) -> dict:
    """
    Count complaints with and without narratives.
    """
    return {
        "with_narrative": df[text_col].notna().sum(),
        "without_narrative": df[text_col].isna().sum()
    }


def get_product_distribution(df: pd.DataFrame, product_col: str) -> pd.Series:
    """
    Get distribution of complaints by product.
    """
    return df[product_col].value_counts()
