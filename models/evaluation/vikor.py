from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluation.entropy_weight import entropy_weights


def vikor_rank(df: pd.DataFrame, v: float = 0.5) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    weights = entropy_weights(numeric).reindex(numeric.columns).fillna(0.0)
    if float(weights.sum()) == 0:
        weights = pd.Series(1 / len(numeric.columns), index=numeric.columns, dtype=float)
    else:
        weights = weights / weights.sum()

    best = numeric.max(axis=0)
    worst = numeric.min(axis=0)
    denom = (best - worst).replace(0, 1)
    regret_matrix = weights * (best - numeric) / denom
    s = regret_matrix.sum(axis=1)
    r = regret_matrix.max(axis=1)
    s_min, s_max = float(s.min()), float(s.max())
    r_min, r_max = float(r.min()), float(r.max())
    s_term = np.zeros(len(s)) if s_max == s_min else (s - s_min) / (s_max - s_min)
    r_term = np.zeros(len(r)) if r_max == r_min else (r - r_min) / (r_max - r_min)
    q = v * s_term + (1 - v) * r_term
    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "vikor_s": s,
            "vikor_r": r,
            "vikor_q": q,
            "rank": pd.Series(q).rank(ascending=True, method="dense").astype(int),
        }
    ).sort_values(["rank", "row_index"]).reset_index(drop=True)
