"""Exploratory Data Analysis (EDA) functions for CFPB complaints dataset."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_product_distribution(df: pd.DataFrame, product_col: str) -> None:
    """
    Plot distribution of complaints by product.
    """
    plt.figure(figsize=(10, 5))
    df[product_col].value_counts().plot(kind="bar")
    plt.title("Complaint Distribution by Product")
    plt.xlabel("Product")
    plt.ylabel("Number of Complaints")
    plt.show()


def analyze_narrative_length(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Compute and plot narrative word count distribution.
    """
    word_counts = df[text_col].astype(str).apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 5))
    sns.histplot(x=word_counts, bins=50)
    plt.title("Narrative Length Distribution (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.show()

    return word_counts.describe()


def count_missing_narratives(df: pd.DataFrame, text_col: str) -> dict:
    """
    Count complaints with and without narratives.
    """
    return {
        "with_narrative": df[text_col].notna().sum(),
        "without_narrative": df[text_col].isna().sum()
    }
