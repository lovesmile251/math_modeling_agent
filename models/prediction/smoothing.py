from __future__ import annotations

import pandas as pd


def smoothing_forecast(df: pd.DataFrame, periods: int = 3, alpha: float = 0.5, window: int = 3) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(values) < 2:
            continue
        exp_value = float(values.iloc[0])
        for value in values.iloc[1:]:
            exp_value = alpha * float(value) + (1 - alpha) * exp_value
        moving_value = float(values.tail(min(window, len(values))).mean())
        for step in range(1, periods + 1):
            rows.append(
                {
                    "target": str(column),
                    "forecast_step": step,
                    "exponential_smoothing": exp_value,
                    "moving_average": moving_value,
                    "alpha": alpha,
                    "window": min(window, len(values)),
                }
            )
    return pd.DataFrame(rows)
