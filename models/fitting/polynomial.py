from __future__ import annotations

import numpy as np
import pandas as pd


def polynomial_fit(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < degree + 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    target_column = _choose_target(numeric)
    x_column = _choose_feature(numeric, target_column)
    if x_column is None:
        return pd.DataFrame()

    x = numeric[x_column].to_numpy(dtype=float)
    y = numeric[target_column].to_numpy(dtype=float)
    if np.allclose(x, x[0]):
        return pd.DataFrame()

    coefficients = np.polyfit(x, y, deg=degree)
    fitted = np.polyval(coefficients, x)
    residual = y - fitted
    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(np.sum(residual**2))
    r_squared = 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss

    rows: list[dict[str, float | str | int]] = []
    for power, coefficient in zip(range(degree, -1, -1), coefficients):
        rows.append(
            {
                "target": str(target_column),
                "feature": str(x_column),
                "degree": int(degree),
                "term": f"x^{power}",
                "coefficient": float(coefficient),
                "r_squared": float(r_squared),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "sample_size": int(len(x)),
            }
        )
    return pd.DataFrame(rows)


def _choose_target(df: pd.DataFrame) -> str:
    priority_keywords = ("target", "y", "demand", "sales", "profit", "revenue", "score")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(df.columns[-1])


def _choose_feature(df: pd.DataFrame, target_column: str) -> str | None:
    candidates = [column for column in df.columns if column != target_column]
    if not candidates:
        return None
    priority_keywords = ("time", "period", "year", "month", "day", "x")
    for column in candidates:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(candidates[0])
