from __future__ import annotations

import numpy as np
import pandas as pd


def linear_regression_summary(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return pd.DataFrame()

    target_column = target_column or _choose_target(numeric)
    feature_columns = [column for column in numeric.columns if column != target_column]
    if not feature_columns:
        return pd.DataFrame()

    y = numeric[target_column].to_numpy(dtype=float)
    x = numeric[feature_columns].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r_squared = 1.0 if total == 0 else 1 - residual / total

    rows = [
        {
            "target": str(target_column),
            "term": "intercept",
            "coefficient": float(coefficients[0]),
            "r_squared": float(r_squared),
            "sample_size": int(len(y)),
        }
    ]
    for column, coefficient in zip(feature_columns, coefficients[1:]):
        rows.append(
            {
                "target": str(target_column),
                "term": str(column),
                "coefficient": float(coefficient),
                "r_squared": float(r_squared),
                "sample_size": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def _choose_target(df: pd.DataFrame) -> str:
    priority_keywords = ("target", "y", "demand", "sales", "profit", "revenue", "score", "需求", "销量", "收益", "得分")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(df.columns[-1])
