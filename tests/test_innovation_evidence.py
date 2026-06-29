from __future__ import annotations

import json

from tools.innovation_evidence import (
    build_innovation_evidence_report,
    innovation_evidence_blocking_issues,
)


def test_innovation_evidence_passes_when_no_claims(project_rooted_workspace):
    report = build_innovation_evidence_report(
        project_rooted_workspace,
        paper_text="This paper reports a baseline forecasting model and result table.",
    )

    assert report["passed"] is True
    assert report["claimed_innovations"] == []
    assert innovation_evidence_blocking_issues(report) == []


def test_innovation_evidence_blocks_unsupported_stacking_claim(project_rooted_workspace):
    report = build_innovation_evidence_report(
        project_rooted_workspace,
        paper_text="Model innovation: we use a Stacking ensemble to improve prediction stability.",
    )

    assert report["passed"] is False
    assert report["claimed_innovations"] == ["stacking_ensemble"]
    assert "stacking_ensemble" in innovation_evidence_blocking_issues(report)[0]


def test_innovation_evidence_accepts_sensitivity_with_ablation_table(project_rooted_workspace):
    (project_rooted_workspace.tables_dir / "feature_ablation.csv").write_text(
        "feature,score\nx1,0.91\n",
        encoding="utf-8",
    )

    report = build_innovation_evidence_report(
        project_rooted_workspace,
        paper_text="The innovation section includes global sensitivity analysis for model robustness.",
    )

    assert report["passed"] is True
    assert report["claimed_innovations"] == ["global_sensitivity_analysis"]


def test_innovation_evidence_prefers_field_level_experiment_proof(project_rooted_workspace):
    report_path = project_rooted_workspace.logs_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(
            {
                "innovation_evidence": {
                    "stacking_ensemble": {
                        "passed": True,
                        "artifacts": ["tables/model_experiment_comparison.csv"],
                        "checks": ["baseline_rmse_improved", "holdout_split_recorded"],
                        "metrics": {"rmse_delta": -0.12},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_innovation_evidence_report(
        project_rooted_workspace,
        paper_text="Model innovation: we use a Stacking ensemble to improve prediction stability.",
    )

    assert report["passed"] is True
    assert report["audits"][0]["proof_source"] == "experiment_report"
    assert report["audits"][0]["proof"]["metrics"]["rmse_delta"] == -0.12


def test_innovation_evidence_rejects_empty_field_level_proof(project_rooted_workspace):
    (project_rooted_workspace.logs_dir / "experiment_report.json").write_text(
        json.dumps(
            {
                "innovation_evidence": [
                    {
                        "innovation_id": "monte_carlo_uncertainty",
                        "passed": True,
                        "artifacts": [],
                        "checks": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_innovation_evidence_report(
        project_rooted_workspace,
        paper_text="The innovation uses Monte Carlo uncertainty quantification.",
    )

    assert report["passed"] is False
    assert report["audits"][0]["proof_source"] == "experiment_report"
