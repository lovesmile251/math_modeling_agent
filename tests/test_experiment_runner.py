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
    assert report["strong_baseline_audit"]["passed"] is False
    assert any(
        "missing executed validation evidence" in issue
        for issue in report["strong_baseline_audit"]["issues"]
    )
    assert report["validation_strategy"] == "rolling_origin_backtest"
    assert set(comparison["role"]) == {"primary", "baseline"}
    assert '"rmse": 2.0' in comparison.loc[
        comparison["model_id"] == "trend_forecast", "metrics_found"
    ].iloc[0]


def test_experiment_runner_strong_baseline_audit_passes_with_validation(temp_workspace):
    source = temp_workspace.data_dir / "sample.csv"
    pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "demand": [100, 112, 121, 135, 150, 166, 181, 199],
            "cost": [30, 31, 34, 36, 39, 41, 44, 47],
        }
    ).to_csv(source, index=False)
    primary_table = temp_workspace.tables_dir / "data_trend_forecast.csv"
    baseline_table = temp_workspace.tables_dir / "data_smoothing_forecast.csv"
    pd.DataFrame({"rmse": [2.0], "mae": [1.5]}).to_csv(primary_table, index=False)
    pd.DataFrame({"rmse": [3.0], "mae": [2.2]}).to_csv(baseline_table, index=False)
    summary = [
        {
            "source": str(source),
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
            ],
        }
    ]
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    plan = ExperimentPlan(
        metrics=["rmse", "mae"],
        validation_strategy="rolling_origin_backtest",
        random_seeds=[42],
    )
    decision = ModelDecision(
        primary_model_id="trend_forecast",
        baseline_model_id="smoothing_forecast",
        selected_model_ids=["trend_forecast", "smoothing_forecast"],
    )

    report_path, _comparison_path = build_experiment_report(temp_workspace, plan, decision)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["strong_baseline_audit"]["passed"] is True
    assert report["executed_validation"].get("rolling_backtest")
    assert report["executed_validation"].get("robustness")
    assert report["executed_validation"].get("ablation")


def test_experiment_runner_accepts_statement_only_baseline_evidence(temp_workspace):
    primary_table = temp_workspace.tables_dir / "statement_trend_forecast.csv"
    baseline_table = temp_workspace.tables_dir / "statement_smoothing_forecast.csv"
    pd.DataFrame(
        {
            "model_id": ["trend_forecast"],
            "role": ["primary"],
            "statement_only_score": [1.0],
        }
    ).to_csv(primary_table, index=False)
    pd.DataFrame(
        {
            "model_id": ["smoothing_forecast"],
            "role": ["baseline"],
            "statement_only_score": [1.0],
        }
    ).to_csv(baseline_table, index=False)
    summary = [
        {
            "source": "statement_only",
            "statement_only": True,
            "model_runs": [
                {
                    "model_id": "trend_forecast",
                    "status": "success",
                    "table": str(primary_table),
                    "elapsed_seconds": 0.0,
                    "mode": "statement_only",
                },
                {
                    "model_id": "smoothing_forecast",
                    "status": "success",
                    "table": str(baseline_table),
                    "elapsed_seconds": 0.0,
                    "mode": "statement_only",
                },
            ],
        }
    ]
    (temp_workspace.logs_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    plan = ExperimentPlan(
        metrics=["statement_only_score"],
        validation_strategy="rolling_origin_backtest",
    )
    decision = ModelDecision(
        primary_model_id="trend_forecast",
        baseline_model_id="smoothing_forecast",
        selected_model_ids=["trend_forecast", "smoothing_forecast"],
    )

    report_path, _comparison_path = build_experiment_report(temp_workspace, plan, decision)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["executed_validation"]["status"] == "statement_only"
    assert report["strong_baseline_audit"]["passed"] is True
    assert (temp_workspace.tables_dir / "statement_baseline_validation.csv").exists()
    assert (temp_workspace.tables_dir / "statement_feature_ablation.csv").exists()
    assert (temp_workspace.tables_dir / "statement_robustness_plan.csv").exists()
