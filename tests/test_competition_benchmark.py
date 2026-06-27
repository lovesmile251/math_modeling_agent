from __future__ import annotations

import json

from agents.base import (
    FormulationSpec,
    ModelDecision,
    ProblemSpec,
    WorkflowState,
)
from tools.competition_benchmark import (
    CompetitionCase,
    evaluate_competition_case,
    load_competition_cases,
)


def test_competition_score_rewards_complete_evidence(temp_workspace):
    state = WorkflowState("test", [], temp_workspace)
    state.problem_spec = ProblemSpec(
        subproblems=[
            {"id": "Q1", "task_type": "forecast"},
            {"id": "Q2", "task_type": "optimization"},
        ]
    )
    state.model_decision = ModelDecision(
        primary_model_id="trend_forecast",
        baseline_model_id="smoothing_forecast",
    )
    state.formulation_spec = FormulationSpec(validation_issues=[])
    experiment = temp_workspace.logs_dir / "experiment_report.json"
    experiment.write_text(
        json.dumps(
            {
                "gate": {"passed": True},
                "executed_validation": {"status": "completed"},
            }
        ),
        encoding="utf-8",
    )
    state.artifacts.update(
        {
            "problem_spec": experiment,
            "model_selection_report": experiment,
            "formulation_spec": experiment,
            "experiment_report": experiment,
            "claim_evidence_map": experiment,
            "paper": experiment,
        }
    )
    state.notes["traceability_coverage_pct"] = "90"
    state.notes["paper_quality_score"] = "85"
    case = CompetitionCase(
        case_id="case",
        title="case",
        expected_task_types=("forecast", "optimization"),
        acceptable_primary_models=("trend_forecast",),
    )

    score = evaluate_competition_case(case, state)

    assert score.total == 100
    assert score.failures == []


def test_benchmark_case_file_is_valid():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    cases = load_competition_cases(
        root / "benchmarks" / "national_competition_cases.json"
    )

    assert len(cases) >= 4
    assert all(case.acceptable_primary_models for case in cases)
