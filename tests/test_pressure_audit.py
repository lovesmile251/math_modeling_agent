from __future__ import annotations

import json
from pathlib import Path

from tools.pressure_audit import build_pressure_audit, format_pressure_audit, write_pressure_audit


def test_pressure_audit_classifies_end_to_end_bottlenecks(tmp_path: Path):
    drill_summary = {
        "results": [
            {
                "case_id": "case-a",
                "workspace": str(tmp_path / "case-a"),
                "elapsed_seconds": 20000,
                "execution_status": "success",
                "execution_attempts": 2,
                "missing_models": ["cvar_optimization"],
                "table_count": 2,
                "figure_count": 1,
                "paper_quality_score": 72,
                "error_count": 1,
                "score": 61.5,
                "artifacts": {"code": "code.py", "paper": "paper.md"},
                "quality_gates": {
                    "export_quality_gate": "failed",
                    "paper_evidence_gate": "failed",
                },
            }
        ]
    }
    contest_summary = {
        "results": [
            {
                "case_id": "case-a",
                "contest_score": 68.0,
                "readiness_band": "not_competitive",
                "answer_correctness_audit": {
                    "applicable": True,
                    "passed": False,
                },
            }
        ]
    }

    audit = build_pressure_audit(
        drill_summary,
        contest_summary,
        time_budget_seconds=21600,
    )
    finding = audit["case_findings"][0]

    assert audit["case_count"] == 1
    assert "model_output_gap" in finding["failure_categories"]
    assert "answer_correctness_failed" in finding["failure_categories"]
    assert "paper_evidence_gate" in finding["failed_gates"]
    assert finding["recommended_phase"] == "code_plan"
    assert audit["failure_taxonomy"]["weak_paper_quality"] == 1
    assert any(item["category"] == "gate:paper_evidence_gate" for item in audit["bottlenecks"])
    clusters = {item["cluster"]: item for item in audit["root_cause_clusters"]}
    assert "execution_and_model_outputs" in clusters
    assert clusters["execution_and_model_outputs"]["recommended_phase"] == "code_plan"
    assert "answer_and_evidence_binding" in clusters
    assert clusters["paper_evidence_and_award_density"]["recommended_phase"] == "section_writing"


def test_pressure_audit_clusters_experiment_and_export_root_causes(tmp_path: Path):
    drill_summary = {
        "results": [
            {
                "case_id": "case-b",
                "workspace": str(tmp_path / "case-b"),
                "elapsed_seconds": 1200,
                "execution_status": "success",
                "missing_models": [],
                "table_count": 14,
                "figure_count": 9,
                "paper_quality_score": 92,
                "error_count": 0,
                "score": 80.0,
                "artifacts": {"code": "code.py", "paper": "paper.md"},
                "quality_gates": {
                    "strong_baseline_gate": "failed",
                    "innovation_evidence_gate": "failed",
                    "export_quality_gate": "passed",
                },
            },
            {
                "case_id": "case-c",
                "workspace": str(tmp_path / "case-c"),
                "elapsed_seconds": 1200,
                "execution_status": "success",
                "missing_models": [],
                "table_count": 14,
                "figure_count": 9,
                "paper_quality_score": 92,
                "error_count": 0,
                "score": 82.0,
                "artifacts": {"code": "code.py"},
                "quality_gates": {},
            },
        ]
    }

    audit = build_pressure_audit(drill_summary)
    clusters = {item["cluster"]: item for item in audit["root_cause_clusters"]}

    assert clusters["experiment_and_innovation_evidence"]["case_count"] == 1
    assert clusters["experiment_and_innovation_evidence"]["gates"] == {
        "innovation_evidence_gate": 1,
        "strong_baseline_gate": 1,
    }
    assert clusters["export_and_gate_coverage"]["case_count"] == 2
    assert clusters["export_and_gate_coverage"]["recommended_phase"] == "export"


def test_pressure_audit_writes_json_and_markdown(tmp_path: Path):
    audit = build_pressure_audit({"results": []})

    paths = write_pressure_audit(audit, tmp_path)
    report = format_pressure_audit(audit)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert "P3 End-to-End Pressure Audit" in report
    assert "Root Cause Clusters" in report
