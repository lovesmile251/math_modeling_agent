from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.base import A_TASK_DELIVERABLE_SPEC, WorkflowState
from agents.evidence_agent import EvidenceAgent
from agents.model_selection_crew import ModelSelectionCrew
from agents.problem_agent import ProblemAgent
from models.optimization.esp import cement_esp_optimization, is_cement_esp_schema
from tools.model_ids import canonical_model_id
from tools.model_registry import registered_model_ids


def _a_case_csv() -> Path | None:
    return next(Path("D:/Desktop").glob("*/Cement_ESP_Data.csv"), None)


def test_problem_agent_writes_task_deliverable_spec(temp_workspace, sample_csv):
    state = WorkflowState(
        problem_text="根据附件数据预测需求趋势，并建立优化模型给出最优方案。",
        data_files=[sample_csv],
        workspace=temp_workspace,
    )

    state = ProblemAgent().run(state)

    assert state.task_deliverable_specs
    assert A_TASK_DELIVERABLE_SPEC in state.artifacts
    payload = json.loads(state.artifacts[A_TASK_DELIVERABLE_SPEC].read_text(encoding="utf-8"))
    assert any(item["task_type"] == "optimization" for item in payload)
    assert any("目标函数值" in item["required_outputs"] for item in payload if item["task_type"] == "optimization")


def test_result_registry_contains_queryable_evidence_records(temp_workspace):
    pd.DataFrame({"score": [1.0, 2.0, 4.0], "cost": [9.0, 8.0, 7.0]}).to_csv(
        temp_workspace.tables_dir / "model_result.csv",
        index=False,
    )
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)

    state = EvidenceAgent().run(state)

    assert state.result_registry is not None
    assert state.result_registry.schema_version == "2.0"
    assert state.result_registry.evidence_records
    assert any(item["kind"] == "column_stat" for item in state.result_registry.evidence_records)
    registry = json.loads(state.artifacts["result_registry"].read_text(encoding="utf-8"))
    assert registry["evidence_records"]
    assert state.claim_evidence_map is not None
    assert state.claim_evidence_map.claims


def test_esp_model_is_in_formal_registry():
    assert "cement_esp_optimization" in registered_model_ids()
    assert canonical_model_id("esp") == "cement_esp_optimization"


def test_cement_esp_optimizer_on_real_a_case_regression():
    csv_path = _a_case_csv()
    if csv_path is None:
        pytest.skip("Cement_ESP_Data.csv is not available on this machine")

    df = pd.read_csv(csv_path)
    assert is_cement_esp_schema(df)

    result = cement_esp_optimization(df)

    assert len(result) == 12
    assert set(result["scenario"]) == {"typical", "high_concentration_high_flow", "low_load"}
    target5 = result[(result["scenario"] == "typical") & (result["target_C_out_mgNm3"] == 5.0)].iloc[0]
    assert bool(target5["constraint_satisfied"]) is True
    assert 4.8 <= float(target5["predicted_C_out_mgNm3"]) <= 5.1
    assert 12.0 <= float(target5["energy_increase_percent"]) <= 15.0
    assert target5["extrapolation_level"] == "strong_extrapolation"


def test_model_selection_prefers_esp_for_real_a_case_schema():
    csv_path = _a_case_csv()
    if csv_path is None:
        pytest.skip("Cement_ESP_Data.csv is not available on this machine")

    problem = "A题：根据水泥窑电除尘 ESP 数据，建立模型优化运行参数，使出口烟尘浓度尽可能低且能耗合理。"
    result = ModelSelectionCrew(llm=None).run(problem, [csv_path], [])
    selected_ids = [item.model_id for item in result.selected]

    assert selected_ids[0] == "cement_esp_optimization"
