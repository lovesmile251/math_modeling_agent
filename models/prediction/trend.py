from __future__ import annotations

import numpy as np
import pandas as pd


def infer_time_column(df: pd.DataFrame) -> str | None:
    keywords = ("year", "date", "time", "month", "period")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return None
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) >= 2 and values.is_monotonic_increasing:
            return str(column)
    return None


def linear_trend_forecast(df: pd.DataFrame, periods: int = 3, time_column: str | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[0] < 2 or numeric.shape[1] == 0:
        return pd.DataFrame()

    time_column = time_column or infer_time_column(df)
    if time_column and time_column in numeric.columns:
        x_source = numeric[time_column]
        target_columns = [column for column in numeric.columns if column != time_column]
    else:
        x_source = pd.Series(np.arange(len(df)), index=df.index)
        target_columns = list(numeric.columns)

    records: list[dict[str, float | str | int]] = []
    for target in target_columns:
        data = pd.DataFrame({"x": x_source, "y": numeric[target]}).dropna()
        if len(data) < 2:
            continue
        x = data["x"].to_numpy(dtype=float)
        y = data["y"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        total = float(np.sum((y - y.mean()) ** 2))
        residual = float(np.sum((y - fitted) ** 2))
        r_squared = 1.0 if total == 0 else 1 - residual / total
        step = _infer_step(x)
        for offset in range(1, periods + 1):
            next_x = float(x.max() + step * offset)
            records.append(
                {
                    "target": str(target),
                    "time_column": time_column or "row_index",
                    "forecast_step": offset,
                    "x": next_x,
                    "forecast": float(slope * next_x + intercept),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                }
            )
    return pd.DataFrame(records)


def _infer_step(values: np.ndarray) -> float:
    unique = np.unique(np.sort(values))
    if len(unique) < 2:
        return 1.0
    diffs = np.diff(unique)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))
