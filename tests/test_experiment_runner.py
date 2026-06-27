from __future__ import annotations

import json

import pandas as pd

from agents.base import ExperimentPlan, ModelDecision
from tools.experiment_runner import build_experiment_report


def test_experiment_runner_compares_primary_and_baseline(temp_workspace):
    primary_table = temp_workspace.tables_dir / "data_trend_forecast.csv"
    baseline_table = temp_workspace.tables_dir / "data_smoothing_forecast.csv"
    pd.DataFrame({"rmse": [2.0], "mae": [1.5]}).to_csv(primary_table, index=False)
    pd.DataFrame({"rmse": [3.0], "mae": [2.2]}).to_csv(baseline_table, index=False)
    summary = [
        {
            "model_runs": [
                {
                    "model_id": "trend_forecast",
                    "status": "success",
                    "table": str(primary_table),
                    "elapsed_seconds": 0.1,
                },
                {
                    "model_id": "smoothing_forecast",
                    "status": "success",
                    "table": str(baseline_table),
                    "elapsed_seconds": 0.1,
                },
            ]
        }
    ]
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    plan = ExperimentPlan(
        metrics=["rmse", "mae"],
        validation_strategy="rolling_origin_backtest",
        random_seeds=[42, 2024],
    )
    decision = ModelDecision(
        primary_model_id="trend_forecast",
        baseline_model_id="smoothing_forecast",
        selected_model_ids=["trend_forecast", "smoothing_forecast"],
    )

    report_path, comparison_path = build_experiment_report(
        temp_workspace, plan, decision
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    assert report["gate"]["passed"] is True
    assert report["validation_strategy"] == "rolling_origin_backtest"
    assert set(comparison["role"]) == {"primary", "baseline"}
    assert '"rmse": 2.0' in comparison.loc[
        comparison["model_id"] == "trend_forecast", "metrics_found"
    ].iloc[0]
