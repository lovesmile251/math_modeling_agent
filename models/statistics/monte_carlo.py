from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_simulation(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    rng = np.random.default_rng(20260620)
    rows = []
    simulations = 5000
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        samples = rng.choice(values, size=simulations, replace=True)
        rows.append(_summary_row(str(column), samples, len(values), "monte_carlo_bootstrap_variable"))

    complete = numeric.dropna()
    if complete.shape[0] > 0 and complete.shape[1] > 1:
        matrix = complete.to_numpy(dtype=float)
        sampled_rows = rng.integers(0, matrix.shape[0], size=simulations)
        portfolio_samples = matrix[sampled_rows].sum(axis=1)
        rows.append(_summary_row("sum_of_numeric_columns", portfolio_samples, int(matrix.shape[0]), "monte_carlo_bootstrap_sum"))

    return pd.DataFrame(rows)


def _summary_row(variable: str, samples: np.ndarray, sample_size: int, method: str) -> dict[str, float | int | str]:
    return {
        "variable": variable,
        "sample_size": sample_size,
        "simulations": int(len(samples)),
        "expected_value": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "p05": float(np.quantile(samples, 0.05)),
        "p50": float(np.quantile(samples, 0.50)),
        "p95": float(np.quantile(samples, 0.95)),
        "probability_positive": float(np.mean(samples > 0)),
        "method": method,
    }
