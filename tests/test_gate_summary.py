from __future__ import annotations

import json

from agents.base import (
    K_EXPORT_BLOCKING_ISSUES,
    K_EXPORT_QUALITY_GATE,
    K_PAPER_EVIDENCE_GATE,
    K_PAPER_EVIDENCE_ISSUES,
    K_PAPER_QUALITY_SCORE,
    WorkflowState,
)
from tools.gate_summary import build_workflow_gate_summary, write_workflow_gate_summary


def test_workflow_gate_summary_records_failed_gates_and_blockers(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXPORT_QUALITY_GATE] = "failed"
    state.notes[K_EXPORT_BLOCKING_ISSUES] = "Submission blocker remains"
    state.notes[K_PAPER_QUALITY_SCORE] = "71"

    summary = build_workflow_gate_summary(state)

    assert summary["paper_quality_score"] == 71
    assert summary["failed_gates"] == [K_EXPORT_QUALITY_GATE]
    assert summary["blockers"][K_EXPORT_BLOCKING_ISSUES] == "Submission blocker remains"
    assert summary["recommended_rework"]["rerun_from_phase"] == "section_writing"

    paths = write_workflow_gate_summary(state)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["failed_gates"] == [K_EXPORT_QUALITY_GATE]
    assert "Workflow Gate Summary" in paths["markdown"].read_text(encoding="utf-8")


def test_workflow_gate_summary_records_nonblocking_rework_recommendation(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_PAPER_QUALITY_SCORE] = "72"

    summary = build_workflow_gate_summary(state)

    assert summary["failed_gates"] == []
    assert summary["recommended_rework"]["can_auto_apply"] is False
    assert summary["recommended_rework"]["rerun_from_phase"] == "section_writing"


def test_workflow_gate_summary_records_paper_evidence_gate(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_PAPER_EVIDENCE_GATE] = "failed"
    state.notes[K_PAPER_EVIDENCE_ISSUES] = "Risk model evidence weak: cvar_optimization"

    summary = build_workflow_gate_summary(state)

    assert K_PAPER_EVIDENCE_GATE in summary["failed_gates"]
    assert summary["blockers"][K_PAPER_EVIDENCE_ISSUES] == "Risk model evidence weak: cvar_optimization"
    assert summary["recommended_rework"]["rerun_from_phase"] == "section_writing"
