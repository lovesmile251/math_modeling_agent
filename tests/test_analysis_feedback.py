from __future__ import annotations

import json

from agents.analysis_agent import AnalysisAgent
from agents.base import WorkflowState


def test_analysis_writes_model_execution_feedback(temp_workspace):
    produced_path = temp_workspace.tables_dir / "sample_trend_forecast.csv"
    empty_path = temp_workspace.tables_dir / "sample_entropy_weights.csv"
    produced_path.write_text("step,value\n1,10\n2,12\n", encoding="utf-8")
    empty_path.write_text("indicator,weight\n", encoding="utf-8")

    summary = [
        {
            "rows": 2,
            "columns": 2,
            "column_names": ["step", "value"],
            "missing_values": {"step": 0, "value": 0},
            "numeric_columns": ["step", "value"],
            "selected_models": ["trend_forecast", "entropy_weights", "topsis_rank"],
            "model_outputs": {
                "trend_forecast": str(produced_path),
                "entropy_weights": str(empty_path),
            },
            "source": "sample.csv",
            "charts": [],
            "describe_table": str(temp_workspace.tables_dir / "sample_describe.csv"),
        }
    ]
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state = AnalysisAgent().run(state)

    feedback_path = temp_workspace.logs_dir / "model_execution_feedback.json"
    assert state.artifacts["model_execution_feedback"] == feedback_path
    payload = json.loads(feedback_path.read_text(encoding="utf-8"))

    assert [item["model_id"] for item in payload["summary"]["produced_models"]] == ["trend_forecast"]
    assert [item["model_id"] for item in payload["summary"]["empty_models"]] == ["entropy_weights"]
    assert [item["model_id"] for item in payload["summary"]["missing_models"]] == ["topsis_rank"]
    assert payload["sources"][0]["produced_models"][0]["rows"] == 2
    assert payload["sources"][0]["empty_models"][0]["reason"] == "empty_table"
    assert payload["sources"][0]["missing_models"][0]["reason"] == "not_found_in_model_outputs"


def test_analysis_accepts_model_runs_summary(temp_workspace):
    produced_path = temp_workspace.tables_dir / "sample_error_analysis.csv"
    produced_path.write_text("metric,value\nRMSE,1.2\n", encoding="utf-8")

    summary = [
        {
            "rows": 2,
            "columns": 2,
            "column_names": ["step", "value"],
            "missing_values": {"step": 0, "value": 0},
            "numeric_columns": ["step", "value"],
            "selected_models": ["error_analysis", "inventory_policy"],
            "model_runs": [
                {
                    "model_id": "error_analysis",
                    "status": "success",
                    "table": str(produced_path),
                    "elapsed_seconds": 0.01,
                    "error": None,
                },
                {
                    "model_id": "inventory_policy",
                    "status": "skipped",
                    "table": None,
                    "elapsed_seconds": 0.01,
                    "error": "empty result",
                },
            ],
            "source": "sample.csv",
            "charts": [],
            "describe_table": str(temp_workspace.tables_dir / "sample_describe.csv"),
        }
    ]
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    AnalysisAgent().run(state)

    payload = json.loads(
        (temp_workspace.logs_dir / "model_execution_feedback.json").read_text(encoding="utf-8")
    )

    assert [item["model_id"] for item in payload["summary"]["produced_models"]] == [
        "error_analysis"
    ]
    assert [item["model_id"] for item in payload["summary"]["missing_models"]] == [
        "inventory_policy"
    ]
