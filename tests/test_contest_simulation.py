from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import tools.contest_simulation as contest_simulation
from tools.contest_simulation import evaluate_contest_readiness, format_contest_report


def test_contest_simulation_scores_first_prize_ready_case():
    drill_summary = {
        "results": [
            {
                "case_id": "toy-a",
                "title": "强交付样例",
                "elapsed_seconds": 1200,
                "execution_status": "success",
                "execution_attempts": 1,
                "produced_models": [
                    "inventory_policy",
                    "error_analysis",
                    "sensitivity_analysis",
                    "model_comparison",
                ],
                "missing_models": [],
                "empty_models": [],
                "table_count": 14,
                "figure_count": 9,
                "paper_quality_score": 95,
                "error_count": 0,
                "artifacts": {
                    "code": "code.py",
                    "paper": "paper.md",
                    "paper_quality": "quality.md",
                    "claim_evidence_map": "claims.json",
                },
                "quality_gates": {
                    "export_quality_gate": "passed",
                    "task_traceability_gate": "passed",
                    "strong_baseline_gate": "passed",
                    "innovation_evidence_gate": "passed",
                    "paper_evidence_gate": "passed",
                },
                "workspace": "workspace/toy-a",
            }
        ]
    }

    summary = evaluate_contest_readiness(drill_summary, time_budget_seconds=6 * 3600)

    assert summary["case_count"] == 1
    assert summary["results"][0]["readiness_band"] == "first_prize_ready"
    assert summary["results"][0]["contest_score"] >= 92
    assert summary["overall_readiness"] == "具备一等奖冲刺工程基础"


def test_contest_simulation_flags_paper_and_model_risks():
    drill_summary = {
        "results": [
            {
                "case_id": "toy-b",
                "title": "风险样例",
                "elapsed_seconds": 21000,
                "execution_status": "success",
                "execution_attempts": 2,
                "produced_models": ["trend_forecast"],
                "missing_models": ["sensitivity_analysis"],
                "empty_models": [],
                "table_count": 2,
                "figure_count": 1,
                "paper_quality_score": 72,
                "error_count": 1,
                "artifacts": {"code": "code.py", "paper": "paper.md"},
                "quality_gates": {
                    "export_quality_gate": "failed",
                    "innovation_evidence_gate": "failed",
                },
                "workspace": "workspace/toy-b",
            }
        ]
    }

    summary = evaluate_contest_readiness(drill_summary, time_budget_seconds=6 * 3600)
    result = summary["results"][0]

    assert result["readiness_band"] in {"risky", "not_competitive"}
    assert any("核心模型未产出" in risk for risk in result["risks"])
    assert any("工作流存在错误记录" in risk for risk in result["risks"])
    assert any("Gate failed" in risk for risk in result["risks"])
    assert result["dimension_scores"]["gate_integrity"] < 10
    assert summary["high_risk_case_count"] == 1


def test_contest_simulation_uses_answer_correctness_gold(tmp_path: Path):
    workspace = tmp_path / "toy-c"
    logs = workspace / "logs"
    tables = workspace / "tables"
    paper = workspace / "paper"
    for path in (logs, tables, paper):
        path.mkdir(parents=True)
    table_path = tables / "answer.csv"
    pd.DataFrame({"metric": ["profit"], "value": [12.0]}).to_csv(table_path, index=False)
    (logs / "result_registry.json").write_text(
        json.dumps({"entries": [{"type": "table", "path": str(table_path)}]}),
        encoding="utf-8",
    )
    (paper / "paper_draft.md").write_text("profit is 12.0", encoding="utf-8")
    drill_summary = {
        "results": [
            {
                "case_id": "toy-c",
                "title": "answer correctness case",
                "elapsed_seconds": 1200,
                "execution_status": "success",
                "execution_attempts": 1,
                "produced_models": [
                    "inventory_policy",
                    "error_analysis",
                    "sensitivity_analysis",
                    "model_comparison",
                ],
                "missing_models": [],
                "empty_models": [],
                "table_count": 14,
                "figure_count": 9,
                "paper_quality_score": 95,
                "error_count": 0,
                "artifacts": {
                    "code": "code.py",
                    "paper": "paper.md",
                    "paper_quality": "quality.md",
                    "claim_evidence_map": "claims.json",
                },
                "quality_gates": {
                    "export_quality_gate": "passed",
                    "task_traceability_gate": "passed",
                    "strong_baseline_gate": "passed",
                    "innovation_evidence_gate": "passed",
                    "paper_evidence_gate": "passed",
                },
                "workspace": str(workspace),
            }
        ]
    }

    summary = evaluate_contest_readiness(
        drill_summary,
        time_budget_seconds=6 * 3600,
        gold_expectations={
            "toy-c": {
                "expected_numeric_ranges": [{"metric": "profit", "min": 40.0, "max": 45.0}]
            }
        },
    )
    result = summary["results"][0]

    assert result["answer_correctness_audit"]["passed"] is False
    assert result["dimension_scores"]["answer_correctness"] == 0
    assert result["readiness_band"] != "first_prize_ready"
    assert any("answer correctness" in risk for risk in result["risks"])


def test_contest_simulation_penalizes_failed_paper_evidence_gate():
    drill_summary = {
        "results": [
            {
                "case_id": "toy-d",
                "title": "paper evidence gate case",
                "elapsed_seconds": 1200,
                "execution_status": "success",
                "execution_attempts": 1,
                "produced_models": [
                    "inventory_policy",
                    "error_analysis",
                    "sensitivity_analysis",
                    "model_comparison",
                ],
                "missing_models": [],
                "empty_models": [],
                "table_count": 14,
                "figure_count": 9,
                "paper_quality_score": 95,
                "error_count": 0,
                "artifacts": {
                    "code": "code.py",
                    "paper": "paper.md",
                    "paper_quality": "quality.md",
                    "claim_evidence_map": "claims.json",
                },
                "quality_gates": {
                    "export_quality_gate": "passed",
                    "task_traceability_gate": "passed",
                    "strong_baseline_gate": "passed",
                    "innovation_evidence_gate": "passed",
                    "paper_evidence_gate": "failed",
                },
                "workspace": "workspace/toy-d",
            }
        ]
    }

    summary = evaluate_contest_readiness(drill_summary, time_budget_seconds=6 * 3600)
    result = summary["results"][0]

    assert result["readiness_band"] != "first_prize_ready"
    assert result["dimension_scores"]["gate_integrity"] < 7
    assert any("paper_evidence_gate" in risk for risk in result["risks"])


def test_contest_simulation_writes_pressure_audit(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_drill(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "case_id": "toy-pressure",
                    "title": "pressure case",
                    "elapsed_seconds": 1200,
                    "execution_status": "success",
                    "execution_attempts": 1,
                    "produced_models": ["trend_forecast"],
                    "missing_models": [],
                    "empty_models": [],
                    "table_count": 1,
                    "figure_count": 0,
                    "paper_quality_score": 70,
                    "error_count": 0,
                    "score": 60.0,
                    "artifacts": {"code": "code.py", "paper": "paper.md"},
                    "quality_gates": {"paper_evidence_gate": "failed"},
                    "workspace": str(tmp_path / "workspace"),
                }
            ]
        }

    monkeypatch.setattr(contest_simulation, "run_real_case_drill", fake_drill)

    contest_simulation.run_contest_simulation(
        corpus_index_path=tmp_path / "index.json",
        corpus_root=tmp_path / "corpus",
        output_dir=tmp_path / "results",
        runs_root=tmp_path / "runs",
        limit=1,
        candidate_profile=True,
    )

    assert captured["export_formats"] == ["docx"]
    assert (tmp_path / "results" / "pressure_audit.json").exists()
    assert (tmp_path / "results" / "pressure_audit.md").exists()
    assert (tmp_path / "results" / "final_delivery_benchmark.json").exists()
    assert (tmp_path / "results" / "final_delivery_benchmark.md").exists()
    final_payload = json.loads(
        (tmp_path / "results" / "final_delivery_benchmark.json").read_text(encoding="utf-8")
    )
    assert final_payload["checks"]["candidate_profile"]["passed"] is True


def test_contest_report_contains_key_metrics():
    summary = {
        "case_count": 1,
        "time_budget_seconds": 21600,
        "average_contest_score": 90.0,
        "average_blind_review_score": 88.0,
        "first_prize_ready_rate": 1.0,
        "high_risk_case_count": 0,
        "overall_readiness": "具备一等奖冲刺工程基础",
        "results": [
            {
                "case_id": "toy-a",
                "contest_score": 94.0,
                "blind_review_score": 90.0,
                "readiness_band": "first_prize_ready",
                "risks": [],
            }
        ],
    }

    report = format_contest_report(summary)

    assert "限时赛制盲审模拟报告" in report
    assert "一等奖就绪率" in report
    assert "toy-a" in report
