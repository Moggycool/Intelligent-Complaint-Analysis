"""Module for stratified sampling of dataframes."""
# from typing import Optional
import pandas as pd
import numpy as np


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    sample_size: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample with proportional representation.

    Guarantees:
    - All classes are represented
    - Sample size is close to requested size
    """

    if sample_size > len(df):
        raise ValueError("Sample size cannot exceed dataset size")

    rng = np.random.default_rng(random_state)

    # Calculate proportions
    label_counts = df[label_col].value_counts()
    proportions = label_counts / label_counts.sum()

    # Initial allocation (rounded)
    allocations = (proportions * sample_size).round().astype(int)

    # Ensure at least 1 sample per class
    allocations[allocations == 0] = 1

    # Adjust total if needed
    diff = sample_size - allocations.sum()

    if diff != 0:
        for label in allocations.index:
            if diff == 0:
                break
            allocations[label] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    # Sample per class
    sampled_parts = []
    for label, n_samples in allocations.items():
        subset = df[df[label_col] == label]
        n_samples = min(n_samples, len(subset))

        # Use the generator to create a reproducible seed for each class sample
        seed = int(rng.integers(0, 2 ** 32 - 1))
        sampled_parts.append(
            subset.sample(n=n_samples, random_state=seed)
        )

    # Shuffle the concatenated sample with a generator-produced seed
    final_seed = int(rng.integers(0, 2 ** 32 - 1))
    return (
        pd.concat(sampled_parts)
        .sample(frac=1, random_state=final_seed)
        .reset_index(drop=True)
    )
