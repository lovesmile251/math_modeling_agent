from __future__ import annotations

import math

import numpy as np
import pandas as pd


def seasonal_decomposition_forecast(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 6 or numeric.shape[1] == 0:
        return pd.DataFrame()

    time_column = _find_column(numeric, ("time", "date", "year", "month", "period", "day", "时间", "日期", "年份", "月份", "周期"))
    targets = [column for column in numeric.columns if str(column) != str(time_column)]
    if not targets:
        targets = list(numeric.columns)

    rows: list[dict[str, float | str | int]] = []
    for target in targets:
        values = pd.to_numeric(numeric[target], errors="coerce").dropna()
        if len(values) < 6:
            continue
        y = values.to_numpy(dtype=float)
        season_length = _infer_season_length(len(y))
        if len(y) < season_length * 2:
            season_length = max(2, len(y) // 2)
        x = np.arange(len(y), dtype=float)
        trend_slope, trend_intercept = np.polyfit(x, y, 1)
        trend = trend_slope * x + trend_intercept
        detrended = y - trend
        seasonal = np.zeros(season_length, dtype=float)
        for index in range(season_length):
            bucket = detrended[np.arange(index, len(detrended), season_length)]
            seasonal[index] = float(bucket.mean()) if len(bucket) else 0.0
        seasonal -= seasonal.mean()
        fitted = trend + seasonal[np.arange(len(y)) % season_length]
        last_x = len(y) - 1
        for step in range(1, periods + 1):
            next_x = last_x + step
            rows.append(
                {
                    "method": "seasonal_decomposition_forecast",
                    "target": str(target),
                    "time_column": str(time_column) if time_column else "row_index",
                    "forecast_step": step,
                    "forecast": float(trend_slope * next_x + trend_intercept + seasonal[next_x % season_length]),
                    "trend_slope": float(trend_slope),
                    "season_length": int(season_length),
                    "rmse": _rmse(y, fitted),
                    "sample_size": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def var_forecast(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 4 or numeric.shape[1] < 2:
        return pd.DataFrame()

    time_column = _find_column(numeric, ("time", "date", "year", "month", "period", "day", "时间", "日期", "年份", "月份", "周期"))
    value_columns = [column for column in numeric.columns if str(column) != str(time_column)]
    if len(value_columns) < 2:
        return pd.DataFrame()

    data = numeric[value_columns].dropna()
    if data.shape[0] < 4 or data.shape[1] < 2:
        return pd.DataFrame()
    values = data.to_numpy(dtype=float)
    x_lag = values[:-1]
    y_now = values[1:]
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    try:
        coefficients = np.linalg.lstsq(design, y_now, rcond=None)[0]
    except np.linalg.LinAlgError:
        return pd.DataFrame()
    fitted = design @ coefficients
    residual_rmse = np.sqrt(np.mean((y_now - fitted) ** 2, axis=0))
    current = values[-1].copy()

    rows: list[dict[str, float | str | int]] = []
    for step in range(1, periods + 1):
        next_value = np.r_[1.0, current] @ coefficients
        for index, column in enumerate(value_columns):
            rows.append(
                {
                    "method": "var_forecast",
                    "target": str(column),
                    "forecast_step": step,
                    "forecast": float(next_value[index]),
                    "lag_order": 1,
                    "intercept": float(coefficients[0, index]),
                    "rmse": float(residual_rmse[index]),
                    "sample_size": int(len(values)),
                }
            )
        current = next_value
    return pd.DataFrame(rows)


def nonlinear_regression_forecast(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 4 or numeric.shape[1] == 0:
        return pd.DataFrame()

    time_column = _find_column(numeric, ("time", "date", "year", "month", "period", "day", "x", "时间", "日期", "年份", "月份", "周期"))
    target_column = _find_column(
        numeric,
        ("target", "y", "demand", "sales", "profit", "revenue", "score", "value", "目标", "需求", "销量", "销售", "利润", "收入", "得分", "值"),
        exclude={time_column} if time_column else set(),
    )
    if target_column is None:
        target_column = _last_numeric_column(numeric, exclude={time_column} if time_column else set())
    if target_column is None:
        return pd.DataFrame()

    data = _xy_data(numeric, time_column, target_column)
    if len(data) < 4:
        return pd.DataFrame()
    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return pd.DataFrame()
    x_shifted = x - x.min()

    candidates: list[dict[str, object]] = []
    try:
        degree = 2 if len(x_shifted) >= 5 else 1
        poly_coefficients = np.polyfit(x_shifted, y, degree)
        fitted = np.polyval(poly_coefficients, x_shifted)
        candidates.append({"model_type": f"polynomial_degree_{degree}", "coefficients": poly_coefficients, "fitted": fitted})
    except (ValueError, np.linalg.LinAlgError):
        pass
    if np.all(y > 0):
        try:
            slope, intercept = np.polyfit(x_shifted, np.log(y), 1)
            fitted = np.exp(intercept + slope * x_shifted)
            candidates.append({"model_type": "exponential", "coefficients": np.array([slope, intercept]), "fitted": fitted})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    if not candidates:
        return pd.DataFrame()

    best = min(candidates, key=lambda item: _rmse(y, item["fitted"]))  # type: ignore[arg-type]
    model_type = str(best["model_type"])
    coefficients = np.asarray(best["coefficients"], dtype=float)
    fitted = np.asarray(best["fitted"], dtype=float)
    step_size = _infer_step(x)
    rows: list[dict[str, float | str | int]] = []
    for step in range(1, periods + 1):
        next_x = float(x.max() + step_size * step)
        next_shifted = next_x - x.min()
        if model_type == "exponential":
            forecast = float(math.exp(coefficients[1] + coefficients[0] * next_shifted))
        else:
            forecast = float(np.polyval(coefficients, next_shifted))
        rows.append(
            {
                "method": "nonlinear_regression_forecast",
                "target": str(target_column),
                "time_column": str(time_column) if time_column else "row_index",
                "forecast_step": step,
                "x": next_x,
                "forecast": forecast,
                "model_type": model_type,
                "rmse": _rmse(y, fitted),
                "r_squared": _r_squared(y, fitted),
                "sample_size": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").dropna(axis=1, how="all").copy()


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    excluded = {str(column) for column in (exclude or set()) if column is not None}
    for column in df.columns:
        if str(column) in excluded:
            continue
        name = str(column).lower()
        if any(_keyword_matches(name, keyword) for keyword in keywords):
            return str(column)
    return None


def _keyword_matches(name: str, keyword: str) -> bool:
    key = keyword.lower()
    if len(key) == 1 and key.isascii() and key.isalpha():
        return name == key
    return key in name


def _last_numeric_column(df: pd.DataFrame, exclude: set[str | None] | None = None) -> str | None:
    excluded = {str(column) for column in (exclude or set()) if column is not None}
    for column in reversed(df.columns):
        if str(column) not in excluded:
            return str(column)
    return None


def _xy_data(df: pd.DataFrame, time_column: str | None, target_column: str) -> pd.DataFrame:
    if time_column and time_column in df.columns:
        data = pd.DataFrame({"x": df[time_column], "y": df[target_column]}).dropna()
    else:
        values = df[target_column].dropna()
        data = pd.DataFrame({"x": np.arange(len(values), dtype=float), "y": values.to_numpy(dtype=float)})
    return data


def _infer_season_length(length: int) -> int:
    if length >= 24:
        return 12
    if length >= 14:
        return 7
    if length >= 8:
        return 4
    return 2


def _infer_step(values: np.ndarray) -> float:
    unique = np.unique(np.sort(values))
    if len(unique) < 2:
        return 1.0
    diffs = np.diff(unique)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))


def _rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    if len(observed) == 0:
        return np.nan
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def _r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total == 0:
        return 1.0
    residual = float(np.sum((observed - fitted) ** 2))
    return float(1.0 - residual / total)
