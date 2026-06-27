from __future__ import annotations

import numpy as np
import pandas as pd


def grey_relation_rank(df: pd.DataFrame, rho: float = 0.5) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    normalized = (numeric - numeric.min()) / (numeric.max() - numeric.min()).replace(0, 1)
    reference = normalized.max(axis=0)
    diff = (normalized - reference).abs()
    min_diff = float(diff.min().min())
    max_diff = float(diff.max().max())
    coefficient = (min_diff + rho * max_diff) / (diff + rho * max_diff)
    grade = coefficient.mean(axis=1)
    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "grey_relation_grade": grade,
            "rank": grade.rank(ascending=False, method="dense").astype(int),
        }
    ).sort_values(["rank", "row_index"]).reset_index(drop=True)
