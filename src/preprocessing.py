"""
Preprocessing module for CFPB complaints dataset.
Includes product normalization, filtering, and narrative cleaning.
"""

import re
from typing import List, Dict
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Download required NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# -----------------------------
# Product normalization mapping
# -----------------------------
PRODUCT_MAP: Dict[str, str] = {
    "credit card": "Credit card",
    "credit cards": "Credit card",
    "credit card or prepaid card": "Credit card",
    "prepaid card": "Credit card",

    # -----------------------------
    # Personal loan variants
    # -----------------------------
    "personal loan": "Personal loan",
    "personal loans": "Personal loan",
    "consumer loan": "Personal loan",
    "payday loan, title loan, or personal loan": "Personal loan",
    "payday loan": "Personal loan",
    "title loan": "Personal loan",

    # -----------------------------
    # Savings account variants
    # -----------------------------
    "savings account": "Savings account",
    "checking or savings account": "Savings account",
    "checking account": "Savings account",
    "bank account or service": "Savings account",

    # -----------------------------
    # Money transfer / money service variants
    # -----------------------------
    "money transfer, virtual currency, or money service": "Money transfers",
    "money transfer": "Money transfers",
    "money transfers": "Money transfers",
    "virtual currency": "Money transfers",
}


# -----------------------------
# Filtering and product normalization
# -----------------------------
def filter_products_and_narratives(
    df: pd.DataFrame,
    product_col: str,
    text_col: str,
    allowed_products: List[str],
    debug: bool = False,
) -> pd.DataFrame:
    """
    Filter dataset to required products and non-empty narratives.

    Args:
        df: Input dataframe
        product_col: Column name for product
        text_col: Column name for consumer complaint narrative
        allowed_products: List of canonical allowed products
        debug: If True, prints diagnostics

    Returns:
        Filtered dataframe
    """
    df = df.copy()
    df[product_col] = df[product_col].astype(str).str.strip()

    # Diagnostics before mapping/filtering
    if debug:
        print("Products (pre-filter) - unique:", df[product_col].nunique())
        print(df[product_col].value_counts().head(10))

    # Create a lowercase-normalized mapping
    normalized_map = {k.strip().lower(): v for k, v in PRODUCT_MAP.items()}

    # Map original product values to canonical names
    df["_product_mapped"] = df[product_col].str.lower().map(normalized_map)

    # Log unmapped products
    if debug:
        unmapped = df[df["_product_mapped"].isna()][product_col].value_counts()
        if len(unmapped) > 0:
            print(
                "Unmapped product sample (will be dropped unless added to PRODUCT_MAP):")
            print(unmapped.head(10))

    # Filter rows: keep only mapped products
    df = df[df["_product_mapped"].notna()]

    # Remove empty or NaN narratives
    df = df[df[text_col].notna() & (df[text_col].str.strip() != "")]

    # Restrict to allowed products (canonical names)
    allowed_canonical = set(allowed_products)
    df = df[df["_product_mapped"].isin(allowed_canonical)]

    # Replace original product column with canonical mapped name and clean up
    df[product_col] = df["_product_mapped"]
    df = df.drop(columns=["_product_mapped"])

    # Diagnostics after filtering
    if debug:
        print("Products (post-filter) - unique:", df[product_col].nunique())
        print(df[product_col].value_counts())

    return df.reset_index(drop=True)


# -----------------------------
# Text cleaning
# -----------------------------
def clean_narrative_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Clean complaint narrative text for embeddings.

    Steps:
    1. Lowercase
    2. Remove boilerplate
    3. Remove special characters
    4. Optional: remove stopwords
    5. Optional: lemmatization
    6. Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"this complaint is regarding",
        r"consumer complaint narrative",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize words
    words = text.split()

    # Remove stopwords if requested
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]

    # Lemmatize if requested
    if lemmatize:
        words = [lemmatizer.lemmatize(w) for w in words]

    # Reconstruct text
    text = " ".join(words)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_text_cleaning(df: pd.DataFrame, text_col: str, debug: bool = False) -> pd.DataFrame:
    """
    Apply narrative text cleaning and remove rows where text is empty after cleaning.

    Args:
        df: Input dataframe
        text_col: Column name for narrative text
        debug: If True, prints diagnostics

    Returns:
        Dataframe with cleaned narratives
    """
    df = df.copy()
    df["cleaned_narrative"] = df[text_col].apply(clean_narrative_text)

    # Remove rows with empty cleaned narratives
    df = df[df["cleaned_narrative"].str.strip() != ""]

    if debug:
        print(f"Rows after cleaning narratives: {len(df)}")

    return df.reset_index(drop=True)


# -----------------------------
# EDA helpers
# -----------------------------
def get_narrative_length_stats(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Get statistics of narrative lengths (word count).
    """
    return df[text_col].astype(str).apply(lambda x: len(x.split())).describe()


def count_narrative_presence(df: pd.DataFrame, text_col: str) -> dict:
    """
    Count complaints with and without narratives.
    """
    return {
        "with_narrative": df[text_col].notna().sum(),
        "without_narrative": df[text_col].isna().sum(),
    }


def get_product_distribution(df: pd.DataFrame, product_col: str) -> pd.Series:
    """
    Get distribution of complaints by product.
    """
    return df[product_col].value_counts()
