from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


_TARGET_KEYWORDS = (
    "target",
    "label",
    "demand",
    "sales",
    "profit",
    "revenue",
    "price",
    "score",
    "value",
    "output",
    "需求",
    "销量",
    "销售",
    "收益",
    "利润",
    "价格",
    "得分",
    "产量",
)

_EXCLUDE_KEYWORDS = ("id", "index", "编号", "序号", "year", "date", "time", "月份", "年份")


def _is_excluded(column: str) -> bool:
    name = str(column).lower()
    return any(keyword in name for keyword in _EXCLUDE_KEYWORDS)


def _choose_target(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    pool = [c for c in columns if not _is_excluded(c)] or columns
    for column in pool:
        if any(keyword in column.lower() for keyword in _TARGET_KEYWORDS):
            return column
    return pool[-1]


def _prepare(df: pd.DataFrame, target_column: str | None, min_rows: int):
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return None
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < min_rows or numeric.shape[1] < 2:
        return None
    target = target_column or _choose_target(numeric)
    if target not in numeric.columns:
        return None
    features = [column for column in numeric.columns if column != target and not _is_excluded(column)]
    if not features:
        features = [column for column in numeric.columns if column != target]
    if not features:
        return None
    y = numeric[target].to_numpy(dtype=float)
    x = numeric[features].to_numpy(dtype=float)
    return target, features, x, y


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def error_analysis(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Residual / goodness-of-fit diagnostics for a linear model of the target.

    Returns a single-row summary frame so downstream readers can pick up the
    metrics directly by column name.
    """
    prepared = _prepare(df, target_column, min_rows=4)
    if prepared is None:
        return pd.DataFrame()
    target, features, x, y = prepared

    coefficients = _fit_linear(x, y)
    predicted = np.column_stack([np.ones(len(x)), x]) @ coefficients
    residual = y - predicted

    n = len(y)
    p = len(features)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_residual = float(np.sum(residual ** 2))
    r_squared = 1.0 if ss_total == 0 else 1 - ss_residual / ss_total
    adjusted = r_squared
    if n - p - 1 > 0:
        adjusted = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    nonzero = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs(residual[nonzero] / y[nonzero])) * 100) if nonzero.any() else float("nan")

    record = {
        "target": str(target),
        "n_samples": int(n),
        "n_features": int(p),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape_percent": mape,
        "max_abs_error": float(np.max(np.abs(residual))),
        "r_squared": float(r_squared),
        "adj_r_squared": float(adjusted),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)) if n > 1 else 0.0,
    }
    return pd.DataFrame([record])


def sensitivity_analysis(
    df: pd.DataFrame,
    target_column: str | None = None,
    perturbation: float = 0.1,
) -> pd.DataFrame:
    """Local sensitivity of a linear model prediction to each input feature.

    Reports the regression coefficient, standardized sensitivity, elasticity and
    a one-at-a-time (OAT) +``perturbation`` response of the predicted target.
    """
    prepared = _prepare(df, target_column, min_rows=4)
    if prepared is None:
        return pd.DataFrame()
    target, features, x, y = prepared

    coefficients = _fit_linear(x, y)
    intercept = float(coefficients[0])
    feature_coefficients = coefficients[1:]
    y_mean = float(np.mean(y))
    y_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    feature_means = x.mean(axis=0)
    feature_stds = x.std(axis=0, ddof=1) if len(y) > 1 else np.zeros(x.shape[1])

    baseline_prediction = intercept + float(np.dot(feature_coefficients, feature_means))

    rows = []
    for index, feature in enumerate(features):
        coefficient = float(feature_coefficients[index])
        x_mean = float(feature_means[index])
        x_std = float(feature_stds[index])
        standardized = coefficient * x_std / y_std if y_std > 1e-12 else 0.0
        elasticity = coefficient * x_mean / y_mean if abs(y_mean) > 1e-12 else 0.0
        delta = x_mean * perturbation if abs(x_mean) > 1e-12 else (x_std * perturbation if x_std > 1e-12 else 1.0)
        perturbed = baseline_prediction + coefficient * delta
        if abs(baseline_prediction) > 1e-12:
            oat_response = (perturbed - baseline_prediction) / abs(baseline_prediction) * 100
        else:
            oat_response = 0.0
        rows.append(
            {
                "feature": str(feature),
                "target": str(target),
                "coefficient": coefficient,
                "standardized_sensitivity": float(standardized),
                "elasticity": float(elasticity),
                f"oat_response_percent_at_{int(perturbation * 100)}pct": float(oat_response),
                "abs_standardized_sensitivity": float(abs(standardized)),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("abs_standardized_sensitivity", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.drop(columns=["abs_standardized_sensitivity"])


def _kfold_indices(n: int, k: int):
    folds = min(k, n)
    indices = np.arange(n)
    return np.array_split(indices, folds)


def model_comparison(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Compare a mean baseline, multiple linear regression and a quadratic fit
    on the same target using K-fold cross-validated RMSE / R²."""
    prepared = _prepare(df, target_column, min_rows=6)
    if prepared is None:
        return pd.DataFrame()
    target, features, x, y = prepared

    correlations = []
    for index in range(x.shape[1]):
        column = x[:, index]
        if np.std(column) == 0:
            correlations.append(0.0)
        else:
            correlations.append(abs(np.corrcoef(column, y)[0, 1]))
    best_feature_index = int(np.argmax(correlations)) if correlations else 0

    folds = _kfold_indices(len(y), k=5)

    def evaluate(predict_builder) -> tuple[float, float]:
        residuals = []
        actuals = []
        for fold in folds:
            test_mask = np.zeros(len(y), dtype=bool)
            test_mask[fold] = True
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 1:
                continue
            predictor = predict_builder(train_mask)
            prediction = predictor(test_mask)
            residuals.extend((y[test_mask] - prediction).tolist())
            actuals.extend(y[test_mask].tolist())
        if not residuals:
            return float("nan"), float("nan")
        residuals_arr = np.array(residuals)
        actuals_arr = np.array(actuals)
        rmse = float(np.sqrt(np.mean(residuals_arr ** 2)))
        ss_total = float(np.sum((actuals_arr - actuals_arr.mean()) ** 2))
        r_squared = float("nan") if ss_total == 0 else float(1 - np.sum(residuals_arr ** 2) / ss_total)
        return rmse, r_squared

    def baseline_builder(train_mask):
        mean_value = float(np.mean(y[train_mask]))
        return lambda test_mask: np.full(test_mask.sum(), mean_value)

    def linear_builder(train_mask):
        design = np.column_stack([np.ones(train_mask.sum()), x[train_mask]])
        coefficients, *_ = np.linalg.lstsq(design, y[train_mask], rcond=None)
        return lambda test_mask: np.column_stack([np.ones(test_mask.sum()), x[test_mask]]) @ coefficients

    def quadratic_builder(train_mask):
        feature = x[train_mask, best_feature_index]
        coefficients = np.polyfit(feature, y[train_mask], 2)
        return lambda test_mask: np.polyval(coefficients, x[test_mask, best_feature_index])

    candidates = [
        ("均值基线 Baseline", baseline_builder),
        ("多元线性回归", linear_builder),
        (f"二次多项式（基于 {features[best_feature_index]}）", quadratic_builder),
    ]

    rows = []
    for name, builder in candidates:
        try:
            rmse, r_squared = evaluate(builder)
        except Exception:
            rmse, r_squared = float("nan"), float("nan")
        rows.append({"target": target, "model": name, "cv_rmse": rmse, "cv_r_squared": r_squared})

    frame = pd.DataFrame(rows)
    valid = frame.dropna(subset=["cv_rmse"])
    if not valid.empty:
        best = valid.sort_values("cv_rmse").iloc[0]["model"]
        frame["is_best_by_rmse"] = frame["model"] == best
    return frame


# ── Model Reviewer ──


@dataclass
class ComparisonResult:
    """Structured result for a single model's validation metrics."""
    model_id: str
    label: str
    task_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    verdict: str = ""  # "best" | "comparable" | "weak"
    notes: list[str] = field(default_factory=list)


@dataclass
class ModelComparisonReport:
    """Aggregated model comparison report across multiple candidates."""
    task_type: str
    candidates: list[ComparisonResult] = field(default_factory=list)
    best_model_id: str = ""
    summary: str = ""
    comparison_csv_ready: list[dict[str, Any]] = field(default_factory=list)


class ModelReviewer:
    """Model reviewer: orchestrates error analysis, sensitivity, and comparison.

    This wraps the existing diagnostics functions and adds task-type-aware
    metric selection so the model selection pipeline can produce a
    ``model_comparison.csv`` and a structured quality report.
    """

    # Metric sets per task type
    METRICS: dict[str, list[str]] = {
        "forecast": ["rmse", "mae", "mape_percent", "r_squared", "adj_r_squared", "max_abs_error"],
        "evaluation": ["ranking_stability", "weight_sensitivity", "spearman_correlation"],
        "optimization": ["objective_value", "constraint_violation", "resource_utilization", "robustness_score"],
        "classification": ["accuracy", "f1_score", "auc", "precision", "recall"],
        "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin"],
        "network": ["modularity", "avg_path_length", "diameter", "clustering_coefficient"],
        "statistics": ["r_squared", "p_value", "effect_size", "confidence_interval_width"],
        "simulation": ["convergence_rate", "monte_carlo_error", "sensitivity_range"],
        "exploration": ["rmse", "mae", "r_squared"],
    }

    def review(
        self,
        df: pd.DataFrame,
        task_type: str = "exploration",
        target_column: str | None = None,
        model_id: str = "",
        label: str = "",
    ) -> ComparisonResult:
        """Run all relevant diagnostics for a single model's output dataframe."""
        result = ComparisonResult(model_id=model_id, label=label, task_type=task_type)

        # Error analysis (always applicable when numeric target exists)
        err_frame = error_analysis(df, target_column)
        if not err_frame.empty:
            row = err_frame.iloc[0].to_dict()
            for key in ("rmse", "mae", "mape_percent", "r_squared", "adj_r_squared", "max_abs_error"):
                val = row.get(key)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    result.metrics[key] = float(val)

        # Sensitivity analysis
        sens_frame = sensitivity_analysis(df, target_column)
        if not sens_frame.empty:
            top_feature = sens_frame.iloc[0]
            result.metrics["top_sensitive_feature"] = str(top_feature.get("feature", ""))
            result.metrics["max_sensitivity"] = float(top_feature.get("standardized_sensitivity", 0))
            result.metrics["sensitivity_range"] = float(
                sens_frame["standardized_sensitivity"].max() - sens_frame["standardized_sensitivity"].min()
            )

        # Model comparison (multi-model benchmark)
        comp_frame = model_comparison(df, target_column)
        if not comp_frame.empty:
            best_row = comp_frame[comp_frame.get("is_best_by_rmse", False)]
            if not best_row.empty:
                result.verdict = "best" if best_row.iloc[0].get("model", "").startswith(label[:4]) else "comparable"
            else:
                result.verdict = "comparable"

        # Default verdict
        if not result.verdict and result.metrics:
            r2 = result.metrics.get("r_squared", 0)
            if r2 > 0.7:
                result.verdict = "best"
            elif r2 > 0.3:
                result.verdict = "comparable"
            else:
                result.verdict = "weak"

        return result

    def generate_comparison_plan(
        self,
        selected_models: list[dict[str, Any]],
        task_type: str = "exploration",
    ) -> ModelComparisonReport:
        """Generate a comparison plan based on the selected models.

        Returns a report that can be serialized to JSON and rendered as CSV.
        """
        report = ModelComparisonReport(task_type=task_type)
        metrics_list = self.METRICS.get(task_type, self.METRICS["exploration"])

        for model in selected_models:
            result = ComparisonResult(
                model_id=model.get("model_id", ""),
                label=model.get("label", ""),
                task_type=task_type,
                metrics={m: 0.0 for m in metrics_list},
            )
            # Pre-fill with expected validation from the model's applicability
            if model.get("validation_plan"):
                result.notes.append(f"计划验证: {model['validation_plan'][0]}")
            report.candidates.append(result)

        # Build CSV-ready rows
        for cand in report.candidates:
            row: dict[str, Any] = {"model_id": cand.model_id, "label": cand.label, "task_type": cand.task_type}
            row.update(cand.metrics)
            report.comparison_csv_ready.append(row)

        if report.candidates:
            report.best_model_id = report.candidates[0].model_id
        report.summary = (
            f"{task_type} 类任务共 {len(report.candidates)} 个候选模型参与对比，"
            f"主要指标: {', '.join(metrics_list[:5])}"
        )
        return report

    def to_comparison_csv(self, report: ModelComparisonReport) -> str:
        """Convert comparison report to CSV string."""
        if not report.comparison_csv_ready:
            return ""
        columns = ["model_id", "label", "task_type"] + list(report.comparison_csv_ready[0].keys() - {"model_id", "label", "task_type"})
        rows = []
        rows.append(",".join(columns))
        for row in report.comparison_csv_ready:
            rows.append(",".join(str(row.get(col, "")) for col in columns))
        return "\n".join(rows)
