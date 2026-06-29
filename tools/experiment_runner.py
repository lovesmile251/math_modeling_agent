from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents.base import ExperimentPlan, ModelDecision
from models.catalog import get_model_contract
from tools.file_tool import write_text
from tools.validation_runner import run_validation_experiments


def build_experiment_report(
    workspace,
    plan: ExperimentPlan | None,
    decision: ModelDecision | None,
) -> tuple[Path, Path]:
    """Create an auditable model-comparison report from actual run artifacts."""
    summary = _load_summary(workspace.logs_dir / "run_summary.json")
    model_runs = _flatten_model_runs(summary)
    plan = plan or ExperimentPlan()
    decision = decision or ModelDecision()
    selected = decision.selected_model_ids or list(model_runs)
    rows: list[dict[str, Any]] = []

    for model_id in selected:
        run = model_runs.get(model_id, {})
        table_path = _resolve_table(workspace.root, run.get("table"))
        table_quality = _table_quality(table_path)
        contract = get_model_contract(model_id)
        extracted_metrics = _extract_metrics(table_path, plan.metrics or list(contract.metrics))
        rows.append(
            {
                "model_id": model_id,
                "role": _role(model_id, decision),
                "status": run.get("status", "not_recorded"),
                "elapsed_seconds": run.get("elapsed_seconds", 0.0),
                "table": str(table_path) if table_path else "",
                **table_quality,
                "metrics_found": json.dumps(extracted_metrics, ensure_ascii=False),
                "contract_metrics": ", ".join(contract.metrics),
                "diagnostics": "; ".join(contract.diagnostics),
            }
        )

    comparison_path = workspace.tables_dir / "model_experiment_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    report = {
        "validation_strategy": plan.validation_strategy,
        "data_split": plan.data_split,
        "test_size": plan.test_size,
        "cv_folds": plan.cv_folds,
        "random_seeds": plan.random_seeds,
        "requested_metrics": plan.metrics,
        "parameter_grid": plan.parameter_grid,
        "sensitivity_plan": plan.sensitivity_plan,
        "ablation_plan": plan.ablation_plan,
        "primary_model_id": decision.primary_model_id,
        "baseline_model_id": decision.baseline_model_id,
        "comparison_table": str(comparison_path),
        "models": rows,
        "gate": _quality_gate(rows, decision),
    }
    source_files = _source_files(summary)
    report["executed_validation"] = run_validation_experiments(
        workspace,
        source_files,
        plan,
        decision,
    )
    report["strong_baseline_audit"] = audit_strong_baseline_evidence(report)
    report_path = write_text(
        workspace.logs_dir / "experiment_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    return report_path, comparison_path


def _load_summary(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _flatten_model_runs(summary: Any) -> dict[str, dict[str, Any]]:
    entries = summary if isinstance(summary, list) else [summary]
    runs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for run in entry.get("model_runs", []):
            if isinstance(run, dict) and run.get("model_id"):
                runs[str(run["model_id"])] = run
    return runs


def _source_files(summary: Any) -> list[Path]:
    entries = summary if isinstance(summary, list) else [summary]
    files: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("source"):
            continue
        path = Path(str(entry["source"]))
        if path.exists():
            files.append(path)
    return files


def _resolve_table(workspace_root: Path, raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = workspace_root / path
    return path if path.exists() else None


def _table_quality(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "table_rows": 0,
            "table_columns": 0,
            "missing_rate": 1.0,
            "finite_numeric_rate": 0.0,
        }
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return {
            "table_rows": 0,
            "table_columns": 0,
            "missing_rate": 1.0,
            "finite_numeric_rate": 0.0,
        }
    cells = max(frame.shape[0] * frame.shape[1], 1)
    numeric = frame.select_dtypes(include="number")
    numeric_cells = max(numeric.size, 1)
    finite = numeric.replace([float("inf"), float("-inf")], pd.NA).notna().sum().sum()
    return {
        "table_rows": int(frame.shape[0]),
        "table_columns": int(frame.shape[1]),
        "missing_rate": round(float(frame.isna().sum().sum() / cells), 6),
        "finite_numeric_rate": round(float(finite / numeric_cells), 6) if not numeric.empty else 1.0,
    }


def _extract_metrics(path: Path | None, requested: list[str]) -> dict[str, float]:
    if path is None:
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return {}
    normalized = {
        str(column).lower().replace("²", "2").replace("_score", ""): column
        for column in frame.columns
    }
    metrics: dict[str, float] = {}
    for metric in requested:
        key = metric.lower().replace("²", "2").replace("_score", "")
        column = normalized.get(key)
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            metrics[metric] = round(float(values.mean()), 8)
    return metrics


def _role(model_id: str, decision: ModelDecision) -> str:
    if model_id == decision.primary_model_id:
        return "primary"
    if model_id == decision.baseline_model_id:
        return "baseline"
    return "candidate"


def _quality_gate(rows: list[dict[str, Any]], decision: ModelDecision) -> dict[str, Any]:
    by_id = {row["model_id"]: row for row in rows}
    issues: list[str] = []
    primary = by_id.get(decision.primary_model_id)
    baseline = by_id.get(decision.baseline_model_id)
    if decision.primary_model_id and (not primary or primary["status"] != "success"):
        issues.append("primary model did not complete successfully")
    if decision.baseline_model_id and (not baseline or baseline["status"] != "success"):
        issues.append("baseline model did not complete successfully")
    if decision.primary_model_id and not decision.baseline_model_id:
        issues.append("no baseline model was designated")
    for row in rows:
        if row["status"] == "success" and row["table_rows"] == 0:
            issues.append(f"{row['model_id']} succeeded without a non-empty result table")
    return {"passed": not issues, "issues": issues}


def audit_strong_baseline_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Audit whether the run has baseline comparison, validation, and ablation evidence."""

    models = report.get("models") if isinstance(report.get("models"), list) else []
    by_role = {
        str(row.get("role")): row
        for row in models
        if isinstance(row, dict) and row.get("role")
    }
    issues: list[str] = []
    evidence: dict[str, Any] = {}

    primary = by_role.get("primary")
    baseline = by_role.get("baseline")
    if not primary:
        issues.append("missing primary model row in experiment comparison")
    elif primary.get("status") != "success" or int(primary.get("table_rows") or 0) <= 0:
        issues.append("primary model lacks a successful non-empty result table")
    if not baseline:
        issues.append("missing baseline model row in experiment comparison")
    elif baseline.get("status") != "success" or int(baseline.get("table_rows") or 0) <= 0:
        issues.append("baseline model lacks a successful non-empty result table")

    comparison_table = str(report.get("comparison_table") or "")
    evidence["comparison_table"] = comparison_table
    if not comparison_table:
        issues.append("missing model experiment comparison table")

    executed = report.get("executed_validation")
    executed = executed if isinstance(executed, dict) else {}
    evidence["executed_validation"] = executed
    validation_keys = {
        key
        for key, value in executed.items()
        if key in {"rolling_backtest", "robustness", "ablation"}
        and value
    }
    if not validation_keys:
        issues.append("missing executed validation evidence")
    if "ablation" not in validation_keys:
        issues.append("missing feature ablation evidence")
    if not ({"rolling_backtest", "robustness"} & validation_keys):
        issues.append("missing strong baseline validation evidence")

    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    if gate.get("passed") is False:
        issues.extend(str(item) for item in gate.get("issues", []) if str(item))

    return {
        "passed": not issues,
        "issues": list(dict.fromkeys(issues)),
        "evidence": evidence,
        "required": [
            "successful primary model",
            "successful baseline model",
            "model experiment comparison table",
            "rolling backtest or perturbation robustness",
            "feature ablation",
        ],
    }
