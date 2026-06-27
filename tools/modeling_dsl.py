from __future__ import annotations

from typing import Any

from agents.base import FormulationSpec, ProblemSpec
from models.catalog import get_model_contract
from tools.model_ids import normalize_model_ids


def build_formulation(
    problem_spec: ProblemSpec | None,
    selected_model_ids: list[str],
) -> FormulationSpec:
    """Build a conservative mathematical formulation from structured tasks."""
    if problem_spec is None:
        return FormulationSpec(validation_issues=["missing structured problem specification"])

    normalized_models = normalize_model_ids(selected_model_ids)
    selected_model_ids = normalized_models.selected

    variables = [
        {
            "name": name,
            "role": _variable_role(name, problem_spec),
            "domain": _infer_domain(name),
            "unit": "unspecified",
        }
        for name in _dedupe(
            [
                *problem_spec.observed_variables,
                *problem_spec.decision_variables,
                *problem_spec.state_variables,
            ]
        )
    ]
    parameters = [
        {"name": name, "source": "data_or_estimation", "unit": "unspecified"}
        for name in problem_spec.parameters
    ]
    stages: list[dict[str, Any]] = []
    for index, task in enumerate(problem_spec.subproblems, start=1):
        task_type = str(task.get("task_type", "exploration"))
        compatible_models = [
            model_id
            for model_id in selected_model_ids
            if task_type in get_model_contract(model_id).task_types
            or task_type == "exploration"
        ]
        stages.append(
            {
                "stage_id": str(task.get("id") or f"S{index}"),
                "task_type": task_type,
                "objective": str(task.get("objective", "")),
                "model_ids": compatible_models,
                "inputs": list(task.get("variables") or problem_spec.observed_variables),
                "outputs": _stage_outputs(task_type),
                "depends_on": [
                    item["from"]
                    for item in problem_spec.task_dependencies
                    if item.get("to") == task.get("id")
                ],
            }
        )

    objectives = _build_objectives(problem_spec)
    constraints = [
        {
            "constraint_id": f"G{index}",
            "expression": constraint,
            "type": _constraint_type(constraint),
            "source": "problem_text",
        }
        for index, constraint in enumerate(problem_spec.constraints, start=1)
    ]
    spec = FormulationSpec(
        variables=variables,
        parameters=parameters,
        objectives=objectives,
        constraints=constraints,
        stages=stages,
        dependencies=problem_spec.task_dependencies,
        assumptions=problem_spec.assumptions,
    )
    spec.validation_issues = validate_formulation(spec)
    if normalized_models.dropped:
        spec.validation_issues.append(
            "dropped unregistered model_id(s): " + ", ".join(normalized_models.dropped)
        )
    return spec


def validate_formulation(spec: FormulationSpec) -> list[str]:
    issues: list[str] = []
    stage_ids = [str(stage.get("stage_id", "")) for stage in spec.stages]
    if len(stage_ids) != len(set(stage_ids)):
        issues.append("stage identifiers must be unique")
    known_stages = set(stage_ids)
    for stage in spec.stages:
        if not stage.get("objective"):
            issues.append(f"{stage.get('stage_id')}: missing objective")
        if not stage.get("model_ids"):
            issues.append(f"{stage.get('stage_id')}: no compatible selected model")
        for dependency in stage.get("depends_on", []):
            if dependency not in known_stages:
                issues.append(f"{stage.get('stage_id')}: unknown dependency {dependency}")
    if any(stage.get("task_type") == "optimization" for stage in spec.stages):
        if not any(variable.get("role") == "decision" for variable in spec.variables):
            issues.append("optimization stage has no declared decision variable")
        if not spec.objectives:
            issues.append("optimization stage has no objective")
    return issues


def _build_objectives(problem_spec: ProblemSpec) -> list[dict[str, str]]:
    objectives: list[dict[str, str]] = []
    for task in problem_spec.subproblems:
        if task.get("task_type") != "optimization":
            continue
        text = f"{task.get('objective', '')} {task.get('source_text', '')}".lower()
        direction = "minimize" if any(
            term in text for term in ("最小", "降低", "成本", "距离", "minimize", "minimum", "cost")
        ) else "maximize"
        objectives.append(
            {
                "objective_id": f"F{len(objectives) + 1}",
                "direction": direction,
                "expression": str(task.get("objective", "objective to be formalized")),
                "source": str(task.get("id", "")),
            }
        )
    return objectives


def _variable_role(name: str, problem_spec: ProblemSpec) -> str:
    if name in problem_spec.decision_variables:
        return "decision"
    if name in problem_spec.state_variables:
        return "state"
    return "observed"


def _infer_domain(name: str) -> str:
    lower = name.lower()
    if any(term in lower for term in ("count", "number", "quantity", "数量", "个数", "班次")):
        return "integer"
    if any(term in lower for term in ("select", "chosen", "是否", "选择")):
        return "binary"
    return "real"


def _stage_outputs(task_type: str) -> list[str]:
    return {
        "forecast": ["forecast_values", "forecast_errors"],
        "evaluation": ["scores", "ranking", "weights"],
        "optimization": ["decision_plan", "objective_value", "constraint_status"],
        "classification": ["predicted_labels", "classification_metrics"],
        "clustering": ["cluster_labels", "cluster_metrics"],
        "network": ["network_metrics", "paths_or_groups"],
        "statistics": ["estimates", "confidence_intervals", "test_results"],
        "simulation": ["simulation_trajectories", "uncertainty_summary"],
    }.get(task_type, ["analysis_results"])


def _constraint_type(text: str) -> str:
    if any(term in text for term in ("不超过", "至多", "<=", "上限")):
        return "upper_bound"
    if any(term in text for term in ("不少于", "至少", ">=", "下限")):
        return "lower_bound"
    if any(term in text for term in ("等于", "必须", "=")):
        return "equality_or_hard"
    return "general"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
