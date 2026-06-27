from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import K_EXECUTION_STATUS, K_PREWRITING_GATE_STATUS, WorkflowState
from models.fitting.advanced import parameter_identification
from models.statistics.sampling import quality_sampling_plan
from tools.pdf_layout_check import check_pdf_render_layout
from tools.real_case_regression import run_real_case_regression
from tools.rework_router import build_rework_plan, recommend_rework_route, write_rework_plan
from tools.exporters import export_pdf
from tools.report_builder import Document, Paragraph


def test_rework_router_sends_execution_failure_to_code_plan(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "failed"

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase.value == "code_plan"
    assert route.severity == "high"


def test_rework_router_builds_executable_plan_file(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "failed"

    plan = build_rework_plan(state)

    assert plan is not None
    assert plan.rerun_from_phase.value == "code_plan"
    assert "execution" in [phase.value for phase in plan.invalidated_phases]
    assert "result_registry" in plan.refresh_artifacts
    plan_path = write_rework_plan(temp_workspace, plan)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["rerun_from_phase"] == "code_plan"
    assert payload["can_auto_apply"] is True


def test_rework_router_sends_prewriting_block_to_result_or_model_phase(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "success"
    state.notes[K_PREWRITING_GATE_STATUS] = "blocked"
    state.notes["prewriting_gate_report"] = "题目或模型选择包含优化任务，但未发现优化结果表。"

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase.value == "model_decision"


def test_pdf_render_layout_check_creates_screenshots(tmp_path):
    pdf = export_pdf(
        Document(title="layout", blocks=[Paragraph(text="正文包含足够内容用于渲染检查。")]),
        tmp_path / "paper.pdf",
    )

    report = check_pdf_render_layout(pdf, tmp_path / "screenshots")

    assert report.pages_checked == 1
    assert report.pages[0].screenshot.endswith(".png")
    assert Path(report.pages[0].screenshot).exists()
    assert report.pages[0].blank is False


def test_quality_sampling_plan_outputs_accept_reject_thresholds():
    df = pd.DataFrame({"sample_count": [100, 120], "defect_count": [8, 10]})

    result = quality_sampling_plan(df)

    assert not result.empty
    assert {"sample_size", "accept_if_defects_leq", "reject_if_defects_geq"}.issubset(result.columns)
    assert (result["sample_size"] > 0).all()


def test_parameter_identification_reports_uncertainty_and_stability():
    df = pd.DataFrame(
        {
            "target": [1.0, 1.3, 1.54, 1.732, 1.8856, 2.00848, 2.106784, 2.1854272],
        }
    )

    result = parameter_identification(df)

    assert not result.empty
    assert {"std_error", "ci95_low", "ci95_high", "stability_indicator", "equation"}.issubset(result.columns)
    assert result["stability_indicator"].iloc[0] == 1
    assert "target[t+1]" in result["equation"].iloc[0]


def test_real_case_regression_runner_handles_missing_corpus_root(tmp_path):
    corpus_index = tmp_path / "corpus.json"
    gold = tmp_path / "gold.json"
    corpus_index.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-a",
                    "year": 2024,
                    "problem": "A",
                    "title": "missing",
                    "statement_path": "missing.docx",
                    "attachment_paths": [],
                    "statement_chars": 0,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-a",
                    "expected_task_types": ["statistics"],
                    "acceptable_primary_models": ["quality_sampling_plan"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_real_case_regression(
        corpus_index_path=corpus_index,
        gold_path=gold,
        corpus_root=tmp_path / "missing-root",
        output_dir=tmp_path / "out",
        limit=20,
    )

    assert summary["requested_cases"] == 1
    assert summary["case_count"] == 0
    assert summary["skipped_count"] == 1
    assert (tmp_path / "out" / "real_case_regression.json").exists()


def test_real_case_regression_runner_executes_tiny_quality_case(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    statement = corpus_root / "case-a.txt"
    statement.write_text(
        "某产品抽样检测需要根据次品率、不合格率设定接收和拒收规则，并给出统计判定方案。",
        encoding="utf-8",
    )
    corpus_index = tmp_path / "corpus.json"
    gold = tmp_path / "gold.json"
    corpus_index.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-a",
                    "year": 2024,
                    "problem": "A",
                    "title": "quality sampling",
                    "statement_path": "case-a.txt",
                    "attachment_paths": [],
                    "statement_chars": 40,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-a",
                    "expected_task_types": ["statistics"],
                    "acceptable_primary_models": ["quality_sampling_plan"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_real_case_regression(
        corpus_index_path=corpus_index,
        gold_path=gold,
        corpus_root=corpus_root,
        output_dir=tmp_path / "out",
        limit=20,
        min_average_score=60.0,
        min_candidate_coverage=1.0,
    )

    assert summary["requested_cases"] == 1
    assert summary["case_count"] == 1
    assert summary["scores"][0]["candidate_hit"] is True
