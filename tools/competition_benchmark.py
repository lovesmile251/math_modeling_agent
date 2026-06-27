from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agents.base import WorkflowState


@dataclass(frozen=True)
class CompetitionCase:
    case_id: str
    title: str
    expected_task_types: tuple[str, ...]
    acceptable_primary_models: tuple[str, ...]
    required_artifacts: tuple[str, ...] = (
        "problem_spec",
        "model_selection_report",
        "formulation_spec",
        "experiment_report",
        "claim_evidence_map",
        "paper",
    )
    min_traceability_pct: float = 70.0
    min_paper_quality: int = 70


@dataclass(frozen=True)
class CompetitionScore:
    case_id: str
    total: float
    dimensions: dict[str, float]
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_competition_case(
    case: CompetitionCase,
    state: WorkflowState,
) -> CompetitionScore:
    """Score one completed workflow without exposing expected answers to agents."""
    failures: list[str] = []
    dimensions: dict[str, float] = {}

    actual_tasks = {
        str(item.get("task_type"))
        for item in (state.problem_spec.subproblems if state.problem_spec else [])
    }
    expected_tasks = set(case.expected_task_types)
    dimensions["task_decomposition"] = (
        20.0 * len(actual_tasks & expected_tasks) / max(len(expected_tasks), 1)
    )
    if not expected_tasks.issubset(actual_tasks):
        failures.append("missing expected task types")

    primary = state.model_decision.primary_model_id if state.model_decision else ""
    dimensions["primary_model"] = (
        20.0 if primary in case.acceptable_primary_models else 0.0
    )
    if dimensions["primary_model"] == 0:
        failures.append(f"unexpected primary model: {primary or 'none'}")

    baseline = state.model_decision.baseline_model_id if state.model_decision else ""
    dimensions["baseline_and_formulation"] = 0.0
    if baseline and state.formulation_spec:
        formulation_issues = state.formulation_spec.validation_issues
        dimensions["baseline_and_formulation"] = 15.0 if not formulation_issues else 8.0
        if formulation_issues:
            failures.append("formulation has validation issues")
    else:
        failures.append("missing baseline or formulation")

    experiment = _read_json_artifact(state, "experiment_report")
    experiment_gate = bool(experiment.get("gate", {}).get("passed"))
    executed_validation = experiment.get("executed_validation", {})
    validation_ran = executed_validation.get("status") == "completed"
    dimensions["experiment_evidence"] = (
        15.0 if experiment_gate and validation_ran else 8.0 if experiment_gate else 0.0
    )
    if not experiment_gate:
        failures.append("experiment gate failed")

    traceability = float(state.notes.get("traceability_coverage_pct", "0") or 0)
    dimensions["traceability"] = min(
        15.0, 15.0 * traceability / max(case.min_traceability_pct, 1)
    )
    if traceability < case.min_traceability_pct:
        failures.append("traceability below threshold")

    quality = int(float(state.notes.get("paper_quality_score", "0") or 0))
    dimensions["paper_quality"] = min(
        10.0, 10.0 * quality / max(case.min_paper_quality, 1)
    )
    if quality < case.min_paper_quality:
        failures.append("paper quality below threshold")

    present = sum(1 for key in case.required_artifacts if key in state.artifacts)
    dimensions["artifact_completeness"] = (
        5.0 * present / max(len(case.required_artifacts), 1)
    )
    if present < len(case.required_artifacts):
        failures.append("required artifacts are incomplete")

    total = round(sum(dimensions.values()), 2)
    return CompetitionScore(
        case_id=case.case_id,
        total=total,
        dimensions={key: round(value, 2) for key, value in dimensions.items()},
        failures=failures,
    )


def load_competition_cases(path: Path) -> list[CompetitionCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        CompetitionCase(
            case_id=str(item["case_id"]),
            title=str(item["title"]),
            expected_task_types=tuple(item["expected_task_types"]),
            acceptable_primary_models=tuple(item["acceptable_primary_models"]),
            required_artifacts=tuple(
                item.get("required_artifacts", CompetitionCase.required_artifacts)
            ),
            min_traceability_pct=float(item.get("min_traceability_pct", 70)),
            min_paper_quality=int(item.get("min_paper_quality", 70)),
        )
        for item in payload
    ]


def _read_json_artifact(state: WorkflowState, key: str) -> dict[str, Any]:
    path = state.artifacts.get(key)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
