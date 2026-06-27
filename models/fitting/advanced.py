from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_least_squares_fit(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_regression(df, allow_single_feature=False)
    if prepared is None:
        return pd.DataFrame()
    x, y, feature_names, target_column, weight_column = prepared

    weights = _extract_weights(df.loc[x.index], weight_column)
    if weights is None:
        weights = np.ones(len(x), dtype=float)
    if len(y) <= len(feature_names) + 1:
        return pd.DataFrame()

    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    sqrt_weights = np.sqrt(weights / max(float(weights.mean()), 1e-12))
    coefficients, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    metrics = _fit_metrics(y, fitted)

    rows: list[dict[str, float | str | int]] = []
    for name, coefficient in zip(["intercept", *feature_names], coefficients):
        rows.append(
            {
                "target": str(target_column),
                "term": str(name),
                "coefficient": float(coefficient),
                "weight_column": str(weight_column) if weight_column is not None else "",
                "r_squared": metrics["r_squared"],
                "rmse": metrics["rmse"],
                "weighted_rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
                "sample_size": int(len(y)),
                "method": "weighted_least_squares",
            }
        )
    return pd.DataFrame(rows)


def nonlinear_least_squares_fit(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_xy(df)
    if prepared is None:
        return pd.DataFrame()
    x, y, feature_column, target_column = prepared
    if len(x) < 5 or np.allclose(x, x[0]):
        return pd.DataFrame()

    candidates = []
    candidates.append(_fit_exponential(x, y))
    candidates.append(_fit_logarithmic(x, y))
    candidates.append(_fit_power(x, y))
    candidates.append(_fit_saturation(x, y))
    valid = [item for item in candidates if item is not None]
    if not valid:
        return pd.DataFrame()

    best_rmse = min(item["rmse"] for item in valid)
    rows: list[dict[str, float | str | int]] = []
    for item in valid:
        rows.append(
            {
                "target": str(target_column),
                "feature": str(feature_column),
                "model": item["model"],
                "parameter_a": float(item.get("a", np.nan)),
                "parameter_b": float(item.get("b", np.nan)),
                "parameter_c": float(item.get("c", np.nan)),
                "r_squared": float(item["r_squared"]),
                "rmse": float(item["rmse"]),
                "selected": int(np.isclose(item["rmse"], best_rmse)),
                "sample_size": int(len(x)),
                "method": "nonlinear_least_squares_candidate_fit",
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "rmse"], ascending=[False, True]).reset_index(drop=True)


def parameter_identification(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 4 or numeric.shape[1] < 1:
        return pd.DataFrame()

    target_column = _choose_target(numeric)
    y_series = numeric[target_column].to_numpy(dtype=float)
    y_next = y_series[1:]
    lagged_y = y_series[:-1]
    exogenous_columns = [column for column in numeric.columns if column != target_column]
    predictors = [lagged_y]
    parameter_names = ["state_feedback"]
    for column in exogenous_columns:
        predictors.append(numeric[column].to_numpy(dtype=float)[:-1])
        parameter_names.append(str(column))

    design = np.column_stack([np.ones(len(y_next)), *predictors])
    if len(y_next) <= design.shape[1]:
        return pd.DataFrame()
    coefficients, *_ = np.linalg.lstsq(design, y_next, rcond=None)
    fitted = design @ coefficients
    residual = y_next - fitted
    metrics = _fit_metrics(y_next, fitted)
    std_errors, ci_low, ci_high = _parameter_uncertainty(design, residual, coefficients)
    phi = float(coefficients[1]) if len(coefficients) > 1 else np.nan
    stable = bool(np.isfinite(phi) and abs(phi) < 1.0)
    time_constant = _discrete_time_constant(phi)
    equation = _dynamic_equation(str(target_column), parameter_names, coefficients)

    rows: list[dict[str, float | str | int]] = [
        {
            "target": str(target_column),
            "parameter": "intercept",
            "estimate": float(coefficients[0]),
            "std_error": float(std_errors[0]),
            "ci95_low": float(ci_low[0]),
            "ci95_high": float(ci_high[0]),
            "r_squared": metrics["r_squared"],
            "rmse": metrics["rmse"],
            "stability_indicator": int(stable),
            "state_feedback_abs": abs(phi) if np.isfinite(phi) else np.nan,
            "discrete_time_constant": time_constant,
            "equation": equation,
            "sample_size": int(len(y_next)),
            "method": "first_order_dynamic_parameter_identification",
        }
    ]
    for index, (name, coefficient) in enumerate(zip(parameter_names, coefficients[1:]), start=1):
        rows.append(
            {
                "target": str(target_column),
                "parameter": str(name),
                "estimate": float(coefficient),
                "std_error": float(std_errors[index]),
                "ci95_low": float(ci_low[index]),
                "ci95_high": float(ci_high[index]),
                "r_squared": metrics["r_squared"],
                "rmse": metrics["rmse"],
                "stability_indicator": int(stable),
                "state_feedback_abs": abs(phi) if np.isfinite(phi) else np.nan,
                "discrete_time_constant": time_constant,
                "equation": equation,
                "sample_size": int(len(y_next)),
                "method": "first_order_dynamic_parameter_identification",
            }
        )
    return pd.DataFrame(rows)


def _prepare_regression(
    df: pd.DataFrame, allow_single_feature: bool
) -> tuple[pd.DataFrame, np.ndarray, list[str], str, str | None] | None:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return None
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    target_column = _choose_target(numeric)
    weight_column = _choose_weight_column(numeric, target_column)
    feature_columns = [column for column in numeric.columns if column not in {target_column, weight_column}]
    if len(feature_columns) < (1 if allow_single_feature else 1):
        return None
    x = numeric[feature_columns]
    if (x.std(ddof=0) > 0).sum() < 1:
        return None
    y = numeric[target_column].to_numpy(dtype=float)
    return x, y, [str(column) for column in feature_columns], str(target_column), str(weight_column) if weight_column is not None else None


def _prepare_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 5 or numeric.shape[1] < 2:
        return None
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    target_column = _choose_target(numeric)
    feature_column = _choose_feature(numeric, target_column)
    if feature_column is None:
        return None
    x = numeric[feature_column].to_numpy(dtype=float)
    y = numeric[target_column].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return x, y, str(feature_column), str(target_column)


def _extract_weights(df: pd.DataFrame, weight_column: str | None) -> np.ndarray | None:
    if weight_column is None or weight_column not in df.columns:
        return None
    weights = pd.to_numeric(df[weight_column], errors="coerce").to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
    if np.isnan(weights).all():
        return None
    fill = float(np.nanmedian(weights))
    return np.where(np.isnan(weights), fill, weights)


def _fit_exponential(x: np.ndarray, y: np.ndarray) -> dict[str, float | str] | None:
    if np.any(y <= 0):
        return None
    design = np.column_stack([np.ones(len(x)), x])
    params, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
    a = float(np.exp(params[0]))
    b = float(params[1])
    fitted = a * np.exp(np.clip(b * x, -50, 50))
    metrics = _fit_metrics(y, fitted)
    return {"model": "exponential_y_equals_a_exp_bx", "a": a, "b": b, **metrics}


def _fit_logarithmic(x: np.ndarray, y: np.ndarray) -> dict[str, float | str] | None:
    shifted_x = x - float(x.min()) + 1e-6 if np.any(x <= 0) else x
    if np.any(shifted_x <= 0):
        return None
    lx = np.log(shifted_x)
    design = np.column_stack([np.ones(len(lx)), lx])
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ params
    metrics = _fit_metrics(y, fitted)
    return {"model": "logarithmic_y_equals_a_plus_b_log_x", "a": float(params[0]), "b": float(params[1]), **metrics}


def _fit_power(x: np.ndarray, y: np.ndarray) -> dict[str, float | str] | None:
    shifted_x = x - float(x.min()) + 1e-6 if np.any(x <= 0) else x
    if np.any(shifted_x <= 0) or np.any(y <= 0):
        return None
    design = np.column_stack([np.ones(len(shifted_x)), np.log(shifted_x)])
    params, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
    a = float(np.exp(params[0]))
    b = float(params[1])
    fitted = a * np.power(shifted_x, b)
    metrics = _fit_metrics(y, fitted)
    return {"model": "power_y_equals_a_x_power_b", "a": a, "b": b, **metrics}


def _fit_saturation(x: np.ndarray, y: np.ndarray) -> dict[str, float | str] | None:
    shifted_x = x - float(x.min()) + 1e-6 if np.any(x <= 0) else x
    if np.any(shifted_x <= 0):
        return None
    candidates = np.linspace(float(np.percentile(shifted_x, 10)), float(np.percentile(shifted_x, 90)), 30)
    best: dict[str, float | str] | None = None
    for b in candidates:
        if b <= 0:
            continue
        basis = shifted_x / (b + shifted_x)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        a = float((basis @ y) / denom)
        fitted = a * basis
        metrics = _fit_metrics(y, fitted)
        item = {"model": "saturation_y_equals_a_x_over_b_plus_x", "a": a, "b": float(b), **metrics}
        if best is None or float(item["rmse"]) < float(best["rmse"]):
            best = item
    return best


def _fit_metrics(y: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    residual = y - fitted
    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(np.sum(residual**2))
    r_squared = 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss
    return {"r_squared": float(r_squared), "rmse": float(np.sqrt(np.mean(residual**2)))}


def _parameter_uncertainty(
    design: np.ndarray,
    residual: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof = max(int(design.shape[0] - design.shape[1]), 1)
    sigma2 = float(np.sum(residual**2) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    std_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if len(std_errors) != len(coefficients):
        std_errors = np.full(len(coefficients), np.nan)
    ci_low = coefficients - 1.96 * std_errors
    ci_high = coefficients + 1.96 * std_errors
    return std_errors, ci_low, ci_high


def _discrete_time_constant(phi: float) -> float:
    if not np.isfinite(phi) or abs(phi) <= 1e-12 or abs(phi) >= 1.0:
        return np.nan
    return float(-1.0 / np.log(abs(phi)))


def _dynamic_equation(target: str, parameter_names: list[str], coefficients: np.ndarray) -> str:
    terms = [f"{coefficients[0]:.6g}"]
    for name, coefficient in zip(parameter_names, coefficients[1:]):
        if name == "state_feedback":
            label = f"{target}[t]"
        else:
            label = f"{name}[t]"
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.6g}*{label}")
    return f"{target}[t+1] = " + " ".join(terms)


def _choose_target(df: pd.DataFrame) -> str:
    priority_keywords = ("target", "y", "demand", "sales", "profit", "revenue", "score", "response")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(df.columns[-1])


def _choose_feature(df: pd.DataFrame, target_column: str) -> str | None:
    candidates = [column for column in df.columns if column != target_column and _choose_weight_column(df[[column, target_column]], target_column) != column]
    if not candidates:
        return None
    priority_keywords = ("time", "period", "year", "month", "day", "x", "input")
    for column in candidates:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(candidates[0])


def _choose_weight_column(df: pd.DataFrame, target_column: str) -> str | None:
    for column in df.columns:
        if column == target_column:
            continue
        name = str(column).lower()
        if name in {"weight", "weights", "w"} or "weight" in name:
            return str(column)
    return None
