from __future__ import annotations

import numpy as np
import pandas as pd


def ahp_weights(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return pd.DataFrame()

    std = numeric.std(numeric_only=True).fillna(0.0)
    if float(std.sum()) == 0:
        weights = pd.Series(1 / len(std), index=std.index, dtype=float)
    else:
        weights = std / std.sum()
    matrix = _pairwise_matrix(weights)
    lambda_max = float(np.mean((matrix @ weights.to_numpy(dtype=float)) / weights.to_numpy(dtype=float)))
    n = len(weights)
    ci = 0.0 if n <= 1 else (lambda_max - n) / (n - 1)
    ri = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri
    return pd.DataFrame(
        {
            "indicator": [str(item) for item in weights.index],
            "weight": weights.to_numpy(dtype=float),
            "lambda_max": lambda_max,
            "consistency_index": ci,
            "consistency_ratio": cr,
            "source": "variance_based_pairwise_matrix",
        }
    )


def _pairwise_matrix(weights: pd.Series) -> np.ndarray:
    values = weights.to_numpy(dtype=float).copy()
    values[values == 0] = 1e-12
    return values[:, None] / values[None, :]
