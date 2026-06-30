from __future__ import annotations

from agents.base import ProblemSpec
from tools.modeling_dsl import build_formulation


def test_formulation_builds_forecast_to_optimization_pipeline():
    problem = ProblemSpec(
        subproblems=[
            {"id": "Q1", "task_type": "forecast", "objective": "预测未来需求"},
            {"id": "Q2", "task_type": "optimization", "objective": "最小化资源成本"},
        ],
        observed_variables=["demand", "cost"],
        decision_variables=["allocation"],
        constraints=["allocation 不超过 capacity"],
        assumptions=["数据口径一致"],
        task_dependencies=[
            {"from": "Q1", "to": "Q2", "reason": "优化使用需求预测"}
        ],
    )

    formulation = build_formulation(
        problem,
        ["trend_forecast", "resource_allocation"],
    )

    assert formulation.validation_issues == []
    assert formulation.stages[1]["depends_on"] == ["Q1"]
    assert formulation.stages[1]["model_ids"] == ["resource_allocation"]
    assert formulation.objectives[0]["direction"] == "minimize"
    assert any(item["role"] == "decision" for item in formulation.variables)


def test_formulation_rejects_optimization_without_decision_variable():
    problem = ProblemSpec(
        subproblems=[
            {"id": "Q1", "task_type": "optimization", "objective": "最大化收益"}
        ],
        observed_variables=["profit"],
    )

    formulation = build_formulation(problem, ["resource_allocation"])

    assert formulation.validation_issues == []
    assert any(variable["name"] == "decision_plan" for variable in formulation.variables)


def test_formulation_deduplicates_conflicting_task_ids_by_modeling_priority():
    problem = ProblemSpec(
        subproblems=[
            {"id": "Q1", "task_type": "classification", "objective": "识别状态"},
            {"id": "Q1", "task_type": "statistics", "objective": "估计参数关系"},
            {"id": "Q2", "task_type": "optimization", "objective": "最小化成本"},
            {"id": "Q2", "task_type": "statistics", "objective": "估计成本参数"},
        ],
        observed_variables=["cost"],
    )

    formulation = build_formulation(problem, ["correlation_analysis", "resource_allocation"])

    assert [stage["stage_id"] for stage in formulation.stages] == ["Q1", "Q2"]
    assert formulation.stages[0]["task_type"] == "statistics"
    assert formulation.stages[1]["task_type"] == "optimization"
    assert formulation.validation_issues == []
