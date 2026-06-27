from __future__ import annotations

import json
from pathlib import Path

import workflows.modeling_workflow as wf
from agents.base import Agent, ErrorCategory, WorkflowError, WorkflowState, format_error_for_display


def _state(temp_workspace) -> WorkflowState:
    return WorkflowState(problem_text="problem", data_files=[], workspace=temp_workspace)


def test_add_error_keeps_legacy_fields_for_string_messages(temp_workspace):
    state = _state(temp_workspace)

    record = state.add_error("manual_agent", "plain failure")

    assert record["agent"] == "manual_agent"
    assert record["message"] == "plain failure"
    assert record["recoverable"] == "True"
    assert record["category"] == "unknown"
    assert record["exception_type"] == "RecordedError"


def test_add_error_classifies_workflow_error_and_details(temp_workspace):
    state = _state(temp_workspace)
    err = WorkflowError(
        "input file is missing",
        category=ErrorCategory.INPUT,
        recoverable=False,
        details={"path": temp_workspace.input_dir / "problem.txt"},
    )

    record = state.add_error("problem_agent", err)

    assert record["category"] == "input"
    assert record["exception_type"] == "WorkflowError"
    assert record["recoverable"] == "False"
    assert record["details"] == {"path": str(temp_workspace.input_dir / "problem.txt")}
    assert format_error_for_display(record) == (
        "[input/WorkflowError] problem_agent: input file is missing"
    )


def test_add_error_classifies_builtin_exceptions(temp_workspace):
    state = _state(temp_workspace)

    record = state.add_error("loader", FileNotFoundError("missing.csv"), recoverable=False)

    assert record["category"] == "io"
    assert record["exception_type"] == "FileNotFoundError"
    assert record["recoverable"] == "False"


class CrashingAgent(Agent):
    name = "crashing_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        raise FileNotFoundError("generated code not found")


def test_workflow_crash_diagnostics_are_structured(monkeypatch, project_rooted_workspace):
    monkeypatch.setattr(wf, "WORKSPACE", project_rooted_workspace)
    workflow = wf.ModelingWorkflow(use_llm=False, skip_review=True, skip_export=True)
    workflow.agents = [CrashingAgent()]

    state = workflow.run("diagnose this")

    assert state.errors
    assert state.errors[0]["category"] == "io"
    assert state.errors[0]["exception_type"] == "FileNotFoundError"

    diagnostics = json.loads(
        Path(project_rooted_workspace.logs_dir / "workflow_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["errors"][0]["category"] == "io"
    assert diagnostics["errors"][0]["exception_type"] == "FileNotFoundError"
