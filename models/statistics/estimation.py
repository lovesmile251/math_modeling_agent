from __future__ import annotations

import math

import numpy as np
import pandas as pd


def parameter_estimation(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if series.empty:
            continue
        values = series.to_numpy(dtype=float)
        mean = float(values.mean())
        variance_mle = float(values.var(ddof=0))
        variance_unbiased = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        std = math.sqrt(max(variance_unbiased, 0.0))
        stderr = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "variable": str(column),
                "sample_size": int(len(values)),
                "mean_mle": mean,
                "variance_mle": variance_mle,
                "variance_unbiased": variance_unbiased,
                "std_unbiased": std,
                "standard_error": stderr,
                "ci95_low_normal_approx": mean - 1.96 * stderr,
                "ci95_high_normal_approx": mean + 1.96 * stderr,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)
