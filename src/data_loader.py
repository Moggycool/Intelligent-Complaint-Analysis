"""Data loader for CFPB complaints dataset."""
from pathlib import Path
import pandas as pd


def load_complaints_csv(path: str | Path) -> pd.DataFrame:
    """
    Load CFPB complaints dataset from CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    return df
