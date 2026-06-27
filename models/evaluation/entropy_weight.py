from __future__ import annotations

import numpy as np
import pandas as pd


def entropy_weights(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.Series(dtype=float)

    normalized = pd.DataFrame(index=numeric.index)
    for column in numeric.columns:
        series = numeric[column].astype(float)
        min_value = series.min()
        max_value = series.max()
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            normalized[column] = 0.0
        else:
            normalized[column] = (series - min_value) / (max_value - min_value)
    normalized = normalized.fillna(0.0)

    if len(normalized) <= 1:
        return pd.Series(1 / len(normalized.columns), index=normalized.columns, dtype=float)

    matrix = normalized.to_numpy(dtype=float) + 1e-12
    column_sums = matrix.sum(axis=0)
    probabilities = matrix / column_sums
    entropy = -np.sum(probabilities * np.log(probabilities), axis=0) / np.log(len(normalized))
    diversity = 1 - entropy
    if np.allclose(diversity.sum(), 0):
        weights = np.ones_like(diversity) / len(diversity)
    else:
        weights = diversity / diversity.sum()
    return pd.Series(weights, index=normalized.columns, dtype=float)
