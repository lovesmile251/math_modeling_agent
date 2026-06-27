from __future__ import annotations

import math

import numpy as np
import pandas as pd


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    rows: list[dict[str, float | str | int]] = []

    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < 2:
            continue
        values = series.to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if std <= 0:
            continue
        t_stat = mean / (std / math.sqrt(len(values)))
        rows.append(
            {
                "test": "one_sample_t_zero",
                "variable": str(column),
                "sample_size": int(len(values)),
                "statistic": float(t_stat),
                "normal_approx_p_value": _two_sided_normal_p(t_stat),
                "null_hypothesis": "mean = 0",
            }
        )

    categorical = df.select_dtypes(exclude="number").dropna(axis=1, how="all")
    for column in categorical.columns:
        counts = df[column].dropna().astype(str).value_counts()
        if len(counts) < 2:
            continue
        expected = counts.sum() / len(counts)
        chi_square = float(((counts - expected) ** 2 / expected).sum()) if expected > 0 else 0.0
        rows.append(
            {
                "test": "chi_square_uniform",
                "variable": str(column),
                "sample_size": int(counts.sum()),
                "statistic": chi_square,
                "normal_approx_p_value": float("nan"),
                "null_hypothesis": "category frequencies are uniform",
            }
        )

    return pd.DataFrame(rows)


def _two_sided_normal_p(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))
