from __future__ import annotations

from agents.base import ProblemSpec
from tools.modeling_dsl import build_formulation, validate_formulation


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

    assert "optimization stage has no declared decision variable" in validate_formulation(
        formulation
    )
