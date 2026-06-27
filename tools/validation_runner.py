from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from agents.base import ExperimentPlan, ModelDecision
from models.prediction.grey_gm import grey_gm11_forecast
from models.prediction.smoothing import smoothing_forecast
from models.prediction.trend import infer_time_column, linear_trend_forecast
from tools.model_registry import ADVANCED_MODEL_REGISTRY, BASIC_MODEL_REGISTRY


_FORECASTERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "trend_forecast": lambda frame: linear_trend_forecast(frame, periods=1),
    "smoothing_forecast": lambda frame: smoothing_forecast(frame, periods=1),
    "grey_gm11": lambda frame: grey_gm11_forecast(frame, periods=1),
}


def run_validation_experiments(
    workspace,
    source_files: list[Path],
    plan: ExperimentPlan,
    decision: ModelDecision,
) -> dict[str, Any]:
    """Execute rolling backtests, perturbation robustness and feature ablation."""
    frames = [_read_table(path) for path in source_files]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return {"status": "skipped", "reason": "no readable source data"}
    frame = frames[0]
    outputs: dict[str, Any] = {"status": "completed"}

    forecast_ids = [
        model_id
        for model_id in decision.selected_model_ids
        if model_id in _FORECASTERS
    ]
    if plan.validation_strategy == "rolling_origin_backtest" and forecast_ids:
        backtest = rolling_origin_backtest(frame, forecast_ids)
        if not backtest.empty:
            path = workspace.tables_dir / "rolling_backtest_metrics.csv"
            backtest.to_csv(path, index=False, encoding="utf-8-sig")
            outputs["rolling_backtest"] = str(path)

    evaluated_ids = list(
        dict.fromkeys(
            item
            for item in (decision.primary_model_id, decision.baseline_model_id)
            if item
        )
    )
    robustness = perturbation_robustness(frame, evaluated_ids, perturbation=0.1)
    if not robustness.empty:
        path = workspace.tables_dir / "model_robustness.csv"
        robustness.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["robustness"] = str(path)

    if decision.primary_model_id:
        ablation = feature_ablation(frame, decision.primary_model_id)
        if not ablation.empty:
            path = workspace.tables_dir / "feature_ablation.csv"
            ablation.to_csv(path, index=False, encoding="utf-8-sig")
            outputs["ablation"] = str(path)
    return outputs


def rolling_origin_backtest(
    frame: pd.DataFrame,
    model_ids: list[str],
    max_origins: int = 8,
) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if len(frame) < 5 or numeric.empty:
        return pd.DataFrame()
    time_column = infer_time_column(frame)
    targets = [
        str(column)
        for column in numeric.columns
        if str(column) != time_column
    ]
    if not targets:
        return pd.DataFrame()
    min_train = max(4, len(frame) // 2)
    origins = list(range(min_train, len(frame)))[-max_origins:]
    records: list[dict[str, Any]] = []

    for model_id in model_ids:
        forecaster = _FORECASTERS[model_id]
        errors: dict[str, list[float]] = {target: [] for target in targets}
        actuals: dict[str, list[float]] = {target: [] for target in targets}
        predictions: dict[str, list[float]] = {target: [] for target in targets}
        for origin in origins:
            result = forecaster(frame.iloc[:origin].copy())
            if result.empty:
                continue
            for target in targets:
                predicted = _forecast_value(result, model_id, target)
                actual = pd.to_numeric(
                    pd.Series([frame.iloc[origin][target]]), errors="coerce"
                ).iloc[0]
                if predicted is None or pd.isna(actual):
                    continue
                predictions[target].append(float(predicted))
                actuals[target].append(float(actual))
                errors[target].append(float(actual) - float(predicted))
        for target in targets:
            if not errors[target]:
                continue
            err = np.asarray(errors[target], dtype=float)
            act = np.asarray(actuals[target], dtype=float)
            pred = np.asarray(predictions[target], dtype=float)
            nonzero = np.abs(act) > 1e-12
            ss_total = float(np.sum((act - act.mean()) ** 2))
            records.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "origins": len(err),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mape": float(np.mean(np.abs(err[nonzero] / act[nonzero])) * 100)
                    if nonzero.any()
                    else float("nan"),
                    "r2": float(1 - np.sum((act - pred) ** 2) / ss_total)
                    if ss_total > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(records)


def perturbation_robustness(
    frame: pd.DataFrame,
    model_ids: list[str],
    perturbation: float = 0.1,
) -> pd.DataFrame:
    numeric_columns = list(frame.select_dtypes(include="number").columns)[:5]
    records: list[dict[str, Any]] = []
    for model_id in model_ids:
        function = _model_callable(model_id)
        if function is None:
            continue
        baseline = _safe_run(function, frame)
        baseline_signature = _output_signature(baseline)
        if baseline_signature is None:
            continue
        for column in numeric_columns:
            for direction in (-1, 1):
                perturbed = frame.copy()
                perturbed[column] = pd.to_numeric(
                    perturbed[column], errors="coerce"
                ) * (1 + direction * perturbation)
                result = _safe_run(function, perturbed)
                signature = _output_signature(result)
                if signature is None:
                    continue
                records.append(
                    {
                        "model_id": model_id,
                        "feature": str(column),
                        "perturbation_pct": direction * perturbation * 100,
                        "baseline_signature": baseline_signature,
                        "perturbed_signature": signature,
                        "relative_change_pct": _relative_change(
                            baseline_signature, signature
                        ),
                    }
                )
    return pd.DataFrame(records)


def feature_ablation(frame: pd.DataFrame, model_id: str) -> pd.DataFrame:
    function = _model_callable(model_id)
    if function is None:
        return pd.DataFrame()
    baseline = _output_signature(_safe_run(function, frame))
    if baseline is None:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    candidate_columns = list(frame.select_dtypes(include="number").columns)[:8]
    for column in candidate_columns:
        if frame.shape[1] <= 1:
            break
        result = _safe_run(function, frame.drop(columns=[column]))
        signature = _output_signature(result)
        records.append(
            {
                "model_id": model_id,
                "removed_feature": str(column),
                "baseline_signature": baseline,
                "ablated_signature": signature,
                "relative_change_pct": _relative_change(baseline, signature)
                if signature is not None
                else float("nan"),
                "status": "success" if signature is not None else "failed_or_empty",
            }
        )
    return pd.DataFrame(records)


def _forecast_value(result: pd.DataFrame, model_id: str, target: str) -> float | None:
    rows = result[result.get("target", pd.Series(dtype=str)).astype(str) == target]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if model_id == "smoothing_forecast":
        values = [
            row.get("exponential_smoothing"),
            row.get("moving_average"),
        ]
        numeric = [float(value) for value in values if pd.notna(value)]
        return float(np.mean(numeric)) if numeric else None
    value = row.get("forecast")
    return float(value) if pd.notna(value) else None


def _model_callable(model_id: str) -> Callable | None:
    basic = BASIC_MODEL_REGISTRY.get(model_id)
    if basic:
        module_name, function_name, _, _ = basic
        return getattr(importlib.import_module(module_name), function_name, None)
    for current_id, _, module_name, function_name, _ in ADVANCED_MODEL_REGISTRY:
        if current_id == model_id:
            return getattr(importlib.import_module(module_name), function_name, None)
    return None


def _safe_run(function: Callable, frame: pd.DataFrame) -> pd.DataFrame:
    try:
        result = function(frame.copy())
    except Exception:
        return pd.DataFrame()
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def _output_signature(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    numeric = frame.select_dtypes(include="number").replace(
        [float("inf"), float("-inf")], np.nan
    )
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float(len(frame))


def _relative_change(baseline: float, changed: float) -> float:
    denominator = max(abs(baseline), 1e-12)
    return float((changed - baseline) / denominator * 100)


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError):
        return None
    return None
