from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.file_tool import write_text


@dataclass(frozen=True)
class PaperEvidenceAudit:
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


RISK_MODEL_METRICS: dict[str, tuple[str, ...]] = {
    "robust_optimization": (
        "robust_value",
        "robust_resource",
        "capacity_slack",
        "uncertainty_rate",
        "robust objective",
        "capacity slack",
    ),
    "scenario_optimization": (
        "expected_value",
        "worst_case_value",
        "max_regret",
        "capacity_slack",
        "expected value",
        "worst-case value",
        "regret",
    ),
    "chance_constrained_optimization": (
        "safe_resource",
        "service_level",
        "feasibility_probability",
        "violation_probability",
        "service level",
        "violation probability",
    ),
    "cvar_optimization": (
        "var_loss",
        "cvar_loss",
        "tail_scenario_count",
        "risk_adjusted_score",
        "expected shortfall",
        "risk-adjusted score",
        "conditional value at risk",
    ),
    "seir_model": (
        "basic_reproduction_number",
        "mean_incubation_period",
        "mean_infectious_period",
        "R0",
        "incubation period",
        "infectious period",
    ),
}


MODEL_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "robust_optimization": ("robust_optimization", "robust optimization", "鲁棒优化", "稳健优化"),
    "scenario_optimization": ("scenario_optimization", "scenario optimization", "情景优化", "场景优化"),
    "chance_constrained_optimization": (
        "chance_constrained_optimization",
        "chance constrained",
        "chance-constrained",
        "机会约束",
        "概率约束",
    ),
    "cvar_optimization": ("cvar_optimization", "CVaR", "tail risk", "尾部风险", "条件风险价值"),
    "seir_model": ("seir_model", "SEIR", "潜伏期", "暴露者"),
}


def audit_paper_evidence_density(text: str, workspace_root: Path | None = None) -> PaperEvidenceAudit:
    issues: list[str] = []
    suggestions: list[str] = []
    metrics: dict[str, int] = {}

    abstract = _extract_section(text, ("摘要", "Abstract"))
    conclusion = _extract_section(text, ("结论", "Conclusion"))
    results = _extract_section(text, ("结果", "Results"))

    abstract_numbers = _substantive_numbers(abstract)
    conclusion_numbers = _substantive_numbers(conclusion)
    result_table_rows = _count_table_rows(results)
    metrics.update(
        {
            "abstract_substantive_numbers": len(abstract_numbers),
            "conclusion_substantive_numbers": len(conclusion_numbers),
            "result_section_table_rows": result_table_rows,
        }
    )

    if abstract and len(abstract_numbers) < 5:
        issues.append(
            "Award evidence density weak: abstract has fewer than 5 substantive numeric result values."
        )
        suggestions.append(
            "Rewrite the abstract around concrete task answers, model metrics, and key quantitative conclusions."
        )

    if results and result_table_rows == 0:
        issues.append("Core result table missing: result section has no Markdown result table.")
        suggestions.append("Add at least one core result table in the result section and interpret it in prose.")

    selected_models = _selected_models_from_report(workspace_root)
    table_stems = _table_stems(workspace_root)
    metrics["selected_models_in_report"] = len(selected_models)
    metrics["generated_result_tables"] = len(table_stems)

    claimed_risk_models = [
        model_id
        for model_id in RISK_MODEL_METRICS
        if _model_is_claimed(text, model_id)
    ]
    metrics["claimed_risk_models"] = len(claimed_risk_models)

    missing_metric_models = []
    for model_id in claimed_risk_models:
        metric_hits = _metric_hits(text, model_id)
        metrics[f"{model_id}_metric_hits"] = metric_hits
        if metric_hits < 2:
            missing_metric_models.append(model_id)
            issues.append(
                f"Risk model evidence weak: {model_id} is claimed without at least 2 model-specific metrics."
            )
            suggestions.append(
                f"For {model_id}, cite its table and discuss: {', '.join(RISK_MODEL_METRICS[model_id][:4])}."
            )

    missing_tables = []
    if workspace_root is not None:
        for model_id in claimed_risk_models:
            if not _matching_model_tables(workspace_root, model_id):
                missing_tables.append(model_id)
    if missing_tables:
        issues.append(
            "Claimed high-level model has no matching generated result table: "
            + ", ".join(missing_tables)
        )
        suggestions.append("Remove unsupported high-level model claims or run the corresponding executable models.")

    selected_risk_models = [model_id for model_id in selected_models if model_id in RISK_MODEL_METRICS]
    metrics["selected_risk_models"] = len(selected_risk_models)
    selected_missing_tables = [
        model_id
        for model_id in selected_risk_models
        if workspace_root is not None and not _matching_model_tables(workspace_root, model_id)
    ]
    metrics["selected_risk_models_missing_tables"] = len(selected_missing_tables)
    if selected_missing_tables:
        issues.append(
            "Selected high-level model has no matching generated result table: "
            + ", ".join(selected_missing_tables[:6])
        )
        suggestions.append(
            "Rerun executable analysis for selected high-level models before writing final claims."
        )
    table_metric_gaps = []
    for model_id in _dedupe([*claimed_risk_models, *selected_risk_models]):
        table_metric_hits = _table_metric_hits(workspace_root, model_id)
        if table_metric_hits is None:
            continue
        metrics[f"{model_id}_table_metric_hits"] = table_metric_hits
        if table_metric_hits == 0:
            table_metric_gaps.append(model_id)
    if table_metric_gaps:
        issues.append(
            "High-level model table lacks model-specific metrics: "
            + ", ".join(table_metric_gaps[:6])
        )
        suggestions.append(
            "Update executable model outputs so each high-level model table exposes its diagnostic metrics."
        )
    missing_narrative = [
        model_id
        for model_id in selected_risk_models
        if not _model_is_claimed(text, model_id)
    ]
    metrics["selected_risk_models_missing_narrative"] = len(missing_narrative)
    if missing_narrative:
        issues.append(
            "Selected high-level model missing from paper narrative: "
            + ", ".join(missing_narrative[:6])
        )
        suggestions.append(
            "Add a model/results paragraph for selected high-level models and cite their table-backed metrics."
        )

    return PaperEvidenceAudit(
        issues=_dedupe(issues),
        suggestions=_dedupe(suggestions),
        metrics=metrics,
    )


def format_paper_evidence_audit(audit: PaperEvidenceAudit) -> str:
    lines = [
        "# Paper Evidence Audit",
        "",
        f"- Gate: {'passed' if audit.passed else 'failed'}",
        f"- Abstract substantive numbers: {audit.metrics.get('abstract_substantive_numbers', 0)}",
        f"- Result-section table rows: {audit.metrics.get('result_section_table_rows', 0)}",
        f"- Claimed high-level risk/mechanism models: {audit.metrics.get('claimed_risk_models', 0)}",
        f"- Generated result tables: {audit.metrics.get('generated_result_tables', 0)}",
        "",
        "## Issues",
    ]
    lines.extend(f"- {item}" for item in (audit.issues or ["None"]))
    lines.extend(["", "## Suggestions"])
    lines.extend(f"- {item}" for item in (audit.suggestions or ["None"]))
    return "\n".join(lines)


def write_paper_evidence_audit(workspace, audit: PaperEvidenceAudit) -> dict[str, Path]:
    json_path = write_text(
        workspace.logs_dir / "paper_evidence_audit.json",
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2),
    )
    md_path = write_text(
        workspace.logs_dir / "paper_evidence_audit.md",
        format_paper_evidence_audit(audit),
    )
    return {"json": json_path, "markdown": md_path}


def _extract_section(text: str, names: tuple[str, ...]) -> str:
    headings = list(re.finditer(r"^#+\s*(.+)$", text, flags=re.MULTILINE))
    for index, heading in enumerate(headings):
        title = heading.group(1)
        if not any(name.lower() in title.lower() for name in names):
            continue
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        return text[start:end].strip()
    return ""


def _substantive_numbers(text: str) -> list[str]:
    values: list[str] = []
    for raw in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?", text):
        try:
            value = float(raw.rstrip("%"))
        except ValueError:
            continue
        if raw.isdigit() and 1900 <= value <= 2100:
            continue
        if raw.isdigit() and 1 <= value <= 12:
            continue
        values.append(raw)
    return values


def _count_table_rows(text: str) -> int:
    return len(re.findall(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE))


def _selected_models_from_report(workspace_root: Path | None) -> list[str]:
    if workspace_root is None:
        return []
    report_path = workspace_root / "logs" / "model_selection_report.json"
    if not report_path.exists():
        return []
    try:
        payload: Any = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    selected = payload.get("selected_model_ids", []) if isinstance(payload, dict) else []
    return [str(item) for item in selected if str(item)]


def _table_stems(workspace_root: Path | None) -> set[str]:
    if workspace_root is None:
        return set()
    tables_dir = workspace_root / "tables"
    if not tables_dir.exists():
        return set()
    return {path.stem.lower() for path in tables_dir.glob("*.csv")}


def _matching_model_tables(workspace_root: Path, model_id: str) -> list[Path]:
    tables_dir = workspace_root / "tables"
    if not tables_dir.exists():
        return []
    suffix = _model_suffix(model_id)
    return [
        path
        for path in tables_dir.glob("*.csv")
        if suffix in path.stem.lower() or model_id in path.stem.lower()
    ]


def _table_metric_hits(workspace_root: Path | None, model_id: str) -> int | None:
    if workspace_root is None:
        return None
    matched_tables = _matching_model_tables(workspace_root, model_id)
    if not matched_tables:
        return None
    metric_tokens = tuple(metric.lower() for metric in RISK_MODEL_METRICS[model_id])
    hits: set[str] = set()
    for path in matched_tables:
        hits.update(_metric_tokens_in_csv(path, metric_tokens))
    return len(hits)


def _metric_tokens_in_csv(path: Path, metric_tokens: tuple[str, ...]) -> set[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(20000)
            handle.seek(0)
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return set()
    haystack = " ".join([*header, sample]).lower()
    return {metric for metric in metric_tokens if metric in haystack}


def _model_is_claimed(text: str, model_id: str) -> bool:
    lowered = text.lower()
    return any(alias.lower() in lowered for alias in MODEL_TEXT_ALIASES.get(model_id, (model_id,)))


def _metric_hits(text: str, model_id: str) -> int:
    lowered = text.lower()
    return sum(1 for metric in RISK_MODEL_METRICS[model_id] if metric.lower() in lowered)


def _model_suffix(model_id: str) -> str:
    return model_id.replace("_model", "").replace("_optimization", "")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
