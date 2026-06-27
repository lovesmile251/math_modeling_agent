from __future__ import annotations

import json

import pandas as pd

from agents.base import WorkflowState
from agents.problem_agent import ProblemAgent


def test_problem_agent_builds_structured_multistage_spec(temp_workspace):
    data_path = temp_workspace.data_dir / "demand.csv"
    pd.DataFrame(
        {
            "month": ["2025-01", "2025-02", "2025-03"],
            "demand": [100, 110, 120],
            "capacity": [105, 115, 125],
            "allocation": [1, 2, 3],
        }
    ).to_csv(data_path, index=False)
    state = WorkflowState(
        problem_text=(
            "问题一：预测未来三个月需求，并使用RMSE评价。"
            "问题二：在容量和预算约束下优化资源分配方案。"
        ),
        data_files=[data_path],
        workspace=temp_workspace,
    )

    state = ProblemAgent().run(state)

    assert state.problem_spec is not None
    assert len(state.problem_spec.subproblems) >= 2
    assert {"forecast", "optimization"}.issubset(
        {item["task_type"] for item in state.problem_spec.subproblems}
    )
    assert "allocation" in state.problem_spec.decision_variables
    assert "RMSE" in state.problem_spec.metrics
    assert state.problem_spec.task_dependencies
    payload = json.loads(
        state.artifacts["problem_spec"].read_text(encoding="utf-8")
    )
    assert payload["subproblems"][0]["id"].startswith("Q")


def test_problem_agent_flags_missing_forecast_time_field(temp_workspace):
    data_path = temp_workspace.data_dir / "data.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(data_path, index=False)
    state = WorkflowState(
        problem_text="预测未来需求。",
        data_files=[data_path],
        workspace=temp_workspace,
    )

    state = ProblemAgent().run(state)

    assert any("时间字段" in issue for issue in state.problem_spec.ambiguities)
