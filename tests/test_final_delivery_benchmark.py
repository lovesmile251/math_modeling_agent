from __future__ import annotations

import json
from pathlib import Path

from tools.final_delivery_benchmark import (
    build_final_delivery_benchmark,
    format_final_delivery_benchmark,
    write_final_delivery_benchmark,
)


def test_final_delivery_benchmark_marks_award_candidate_when_all_checks_pass():
    contest = {
        "case_count": 3,
        "average_contest_score": 94.0,
        "average_blind_review_score": 90.0,
        "first_prize_ready_rate": 0.67,
        "high_risk_case_count": 0,
        "run_profile": {"candidate_profile": True},
        "results": [
            {"case_id": "a", "readiness_band": "first_prize_ready", "risks": []},
            {"case_id": "b", "readiness_band": "first_prize_ready", "risks": []},
            {"case_id": "c", "readiness_band": "competitive", "risks": []},
        ],
    }
    pressure = {
        "aggregate": {
            "export_ready_rate": 1.0,
            "paper_evidence_pass_rate": 1.0,
        },
        "root_cause_clusters": [],
        "case_findings": [],
    }

    summary = build_final_delivery_benchmark(contest, pressure)

    assert summary["status"] == "award_candidate"
    assert summary["passed"] is True
    assert summary["failed_checks"] == []


def test_final_delivery_benchmark_blocks_smoke_profile_even_with_high_scores():
    contest = {
        "case_count": 2,
        "average_contest_score": 95.0,
        "average_blind_review_score": 91.0,
        "first_prize_ready_rate": 1.0,
        "high_risk_case_count": 0,
        "run_profile": {"candidate_profile": False},
        "results": [
            {"case_id": "a", "readiness_band": "first_prize_ready", "risks": []},
            {"case_id": "b", "readiness_band": "first_prize_ready", "risks": []},
        ],
    }
    pressure = {
        "aggregate": {
            "export_ready_rate": 1.0,
            "paper_evidence_pass_rate": 1.0,
        },
        "root_cause_clusters": [],
        "case_findings": [],
    }

    summary = build_final_delivery_benchmark(contest, pressure)

    assert summary["status"] == "not_ready"
    assert "candidate_profile" in summary["failed_checks"]
    assert any("--candidate-profile" in item for item in summary["recommendations"])


def test_final_delivery_benchmark_blocks_low_scores_and_failed_gates():
    contest = {
        "case_count": 2,
        "average_contest_score": 72.0,
        "average_blind_review_score": 68.0,
        "first_prize_ready_rate": 0.0,
        "high_risk_case_count": 1,
        "run_profile": {"candidate_profile": True},
        "results": [
            {
                "case_id": "weak",
                "readiness_band": "not_competitive",
                "risks": ["answer correctness expectations failed"],
            }
        ],
    }
    pressure = {
        "aggregate": {
            "export_ready_rate": 0.5,
            "paper_evidence_pass_rate": 0.5,
        },
        "root_cause_clusters": [
            {"cluster": "execution_and_model_outputs", "priority": 0}
        ],
        "case_findings": [
            {"case_id": "weak", "failed_gates": ["paper_evidence_gate"]}
        ],
    }

    summary = build_final_delivery_benchmark(contest, pressure)

    assert summary["status"] == "not_ready"
    assert summary["passed"] is False
    assert "average_contest_score" in summary["failed_checks"]
    assert "p0_root_cause_clusters" in summary["failed_checks"]
    assert any("failed gates" in finding for finding in summary["blocking_findings"])
    assert summary["recommendations"]


def test_final_delivery_benchmark_writes_reports(tmp_path: Path):
    summary = build_final_delivery_benchmark(
        {
            "case_count": 0,
            "average_contest_score": 0,
            "average_blind_review_score": 0,
            "first_prize_ready_rate": 0,
            "high_risk_case_count": 0,
            "results": [],
        },
        {"aggregate": {}, "root_cause_clusters": [], "case_findings": []},
    )

    paths = write_final_delivery_benchmark(summary, tmp_path)
    report = format_final_delivery_benchmark(summary)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert "Final Delivery Benchmark" in report
