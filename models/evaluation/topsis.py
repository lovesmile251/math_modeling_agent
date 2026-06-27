from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluation.entropy_weight import entropy_weights


def topsis_rank(df: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] == 0 or numeric.shape[1] == 0:
        return pd.DataFrame()

    if weights is None:
        weights = entropy_weights(numeric)
    weights = weights.reindex(numeric.columns).fillna(0.0)
    if weights.sum() == 0:
        weights = pd.Series(1 / len(numeric.columns), index=numeric.columns, dtype=float)
    else:
        weights = weights / weights.sum()

    matrix = numeric.to_numpy(dtype=float)
    norms = np.sqrt((matrix**2).sum(axis=0))
    norms[norms == 0] = 1.0
    weighted = matrix / norms * weights.to_numpy(dtype=float)
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    denominator = distance_best + distance_worst
    score = np.divide(distance_worst, denominator, out=np.zeros_like(distance_worst), where=denominator != 0)

    result = pd.DataFrame(
        {
            "row_index": numeric.index,
            "topsis_score": score,
            "rank": pd.Series(score).rank(ascending=False, method="dense").astype(int),
        }
    )
    return result.sort_values(["rank", "row_index"]).reset_index(drop=True)
