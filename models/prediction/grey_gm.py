from __future__ import annotations

import numpy as np
import pandas as pd


def grey_gm11_forecast(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 4:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < 4 or (series <= 0).any():
            continue
        x0 = series.to_numpy(dtype=float)
        x1 = np.cumsum(x0)
        z1 = 0.5 * (x1[1:] + x1[:-1])
        b = np.column_stack([-z1, np.ones(len(z1))])
        y = x0[1:]
        try:
            a, u = np.linalg.lstsq(b, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if a == 0:
            continue
        fitted_acc = [(x0[0] - u / a) * np.exp(-a * k) + u / a for k in range(len(x0) + periods + 1)]
        predicted = np.diff(fitted_acc)
        for step in range(1, periods + 1):
            rows.append(
                {
                    "target": str(column),
                    "forecast_step": step,
                    "forecast": float(predicted[len(x0) + step - 1]),
                    "a": float(a),
                    "u": float(u),
                }
            )
    return pd.DataFrame(rows)
