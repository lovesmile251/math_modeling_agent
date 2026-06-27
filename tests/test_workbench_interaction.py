from __future__ import annotations

from agents.base import PaperOutline, PhaseStatus, ReviewFindings, WorkflowPhase, WorkflowState
from app.streamlit_app import (
    _feedback_templates_for_phase,
    _next_user_action,
    _phase_completion_ratio,
    _phase_status_counts,
    _revision_target_options,
    _split_csv_items,
)


def test_phase_counts_and_next_action_detect_waiting_checkpoint(temp_workspace):
    state = WorkflowState(problem_text="problem", data_files=[], workspace=temp_workspace)
    state.set_phase_status(WorkflowPhase.PROBLEM_ANALYSIS, PhaseStatus.COMPLETED)
    state.set_phase_status(WorkflowPhase.MODEL_DECISION, PhaseStatus.WAITING_FOR_USER)

    counts = _phase_status_counts(state)

    assert counts[PhaseStatus.COMPLETED.value] == 1
    assert counts[PhaseStatus.WAITING_FOR_USER.value] == 1
    assert _phase_completion_ratio(state) > 0
    assert _next_user_action(state) == f"确认或返工：{WorkflowPhase.MODEL_DECISION.label}"


def test_revision_targets_include_outline_sections_and_review_findings(temp_workspace):
    state = WorkflowState(problem_text="problem", data_files=[], workspace=temp_workspace)
    state.paper_outline = PaperOutline(
        sections=[
            {"id": "abstract", "title": "摘要", "evidence_ids": ["E1"]},
            {"id": "model", "title": "模型建立", "evidence_ids": ["E2"]},
        ],
        total_sections=2,
    )
    state.review_findings = ReviewFindings(
        reviewer="language",
        issues=[{"id": "tone", "title": "语言偏口语"}],
    )

    options = _revision_target_options(state)

    assert options["paper_section:abstract"] == "论文章节：摘要"
    assert options["paper_section:model"] == "论文章节：模型建立"
    assert options["review_finding:tone"] == "审稿问题：语言偏口语"


def test_feedback_templates_and_csv_split_are_phase_aware():
    templates = _feedback_templates_for_phase(WorkflowPhase.RESULT_ANALYSIS)

    assert "强化结论证据" in templates
    assert _feedback_templates_for_phase(WorkflowPhase.PROBLEM_ANALYSIS) == {}
    assert _split_csv_items(" E1, ,E2,E3 ") == ["E1", "E2", "E3"]
