from __future__ import annotations

from agents.base import (
    K_EXECUTION_STATUS,
    K_LLM_FAILURE_KIND,
    K_PAPER_QUALITY_SCORE,
    K_RESULT_ANALYSIS,
    K_SELECTED_MODEL_IDS,
    ProblemSpec,
    WorkflowState,
)
from tools.model_ids import canonical_model_id, normalize_model_ids
from tools.modeling_dsl import build_formulation
from tools.prewriting_gate import evaluate_pre_writing_gate
from workflows.modeling_workflow import ModelingWorkflow


def test_model_id_normalization_accepts_common_aliases():
    result = normalize_model_ids(["correlation-analysis", "Pearson correlation", "not_a_model"])

    assert result.selected == ["correlation_analysis"]
    assert result.dropped == ["not_a_model"]
    assert canonical_model_id("TOPSIS") == "topsis_rank"


def test_formulation_drops_unknown_model_ids_without_crashing():
    problem = ProblemSpec(
        subproblems=[{"id": "Q1", "task_type": "statistics", "objective": "检验变量相关性"}],
        observed_variables=["x", "y"],
    )

    formulation = build_formulation(problem, ["correlation-analysis", "ghost_model"])

    assert formulation.stages[0]["model_ids"] == ["correlation_analysis"]
    assert any("ghost_model" in issue for issue in formulation.validation_issues)


def test_pre_writing_gate_blocks_optimization_without_result_tables(temp_workspace):
    state = WorkflowState(
        problem_text="请建立优化模型，最小化能耗并给出最优控制参数。",
        data_files=[],
        workspace=temp_workspace,
    )
    state.notes[K_EXECUTION_STATUS] = "success"
    state.notes[K_SELECTED_MODEL_IDS] = '["nonlinear_optimization"]'
    state.notes[K_RESULT_ANALYSIS] = "结果分析" * 80

    report = evaluate_pre_writing_gate(state)

    assert report.ok is False
    assert any("优化结果表" in issue for issue in report.issues)


def test_workflow_skips_writing_retry_after_quota_failure(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "success"
    state.notes[K_PAPER_QUALITY_SCORE] = "20"
    state.notes[K_LLM_FAILURE_KIND] = "quota"

    workflow = ModelingWorkflow(use_llm=False, workspace=temp_workspace)

    class DummyWritingAgent:
        name = "writing_agent"
        calls = 0

        def run(self, state):
            self.calls += 1
            return state

    class DummyReviewAgent:
        name = "review_agent"

        def run(self, state):
            return state

    writing = DummyWritingAgent()
    workflow.agents = [writing, DummyReviewAgent()]

    workflow._retry_writing_review(state, __import__("logging").getLogger("test"))

    assert writing.calls == 0
    assert state.notes["writing_retry_stop_reason"] == "non-retryable LLM failure: quota"
