from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import (
    K_AUTO_REWORK_REPAIR_BRIEF,
    K_AUTO_REWORK_REPAIR_HINTS,
    K_EXECUTION_STATUS,
    K_PREWRITING_GATE_STATUS,
    PhaseStatus,
    WorkflowPhase,
    WorkflowState,
)
from agents.model_selection_crew import ModelSelectionCrew
from models.fitting.advanced import parameter_identification
from models.optimization.planting import crop_planting_plan
from models.statistics.nipt import nipt_bmi_grouping
from models.statistics.sampling import quality_sampling_plan
from tools.pdf_layout_check import check_pdf_render_layout
from tools.real_case_regression import run_real_case_regression
from tools.rework_router import apply_rework_plan, build_rework_plan, recommend_rework_route, write_rework_plan
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
    assert any("non-empty CSV" in hint for hint in plan.repair_hints)
    plan_path = write_rework_plan(temp_workspace, plan)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["rerun_from_phase"] == "code_plan"
    assert payload["can_auto_apply"] is True
    assert payload["repair_hints"]


def test_rework_plan_application_marks_downstream_state(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "failed"
    state.artifacts["result_registry"] = temp_workspace.logs_dir / "result_registry.json"

    plan = build_rework_plan(state)
    result = apply_rework_plan(state, plan, clear_artifacts=True)

    assert result.applied is True
    assert result.rerun_from_phase == WorkflowPhase.CODE_PLAN
    assert state.phase == WorkflowPhase.CODE_PLAN
    assert state.get_phase_status(WorkflowPhase.CODE_PLAN) == PhaseStatus.NEEDS_REVISION
    assert state.get_phase_status(WorkflowPhase.EXECUTION) == PhaseStatus.NEEDS_REVISION
    assert "result_registry" in result.stale_artifacts
    assert "result_registry" in result.removed_artifacts
    assert "result_registry" not in state.artifacts
    assert state.notes[K_AUTO_REWORK_REPAIR_HINTS]
    assert "non-empty CSV" in state.notes[K_AUTO_REWORK_REPAIR_BRIEF]
    assert state.decisions[-1]["action"] == "apply_rework_plan"


def test_rework_router_sends_prewriting_block_to_result_or_model_phase(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes[K_EXECUTION_STATUS] = "success"
    state.notes[K_PREWRITING_GATE_STATUS] = "blocked"
    state.notes["prewriting_gate_report"] = "题目或模型选择包含优化任务，但未发现优化结果表。"

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase.value == "model_decision"


def test_rework_router_sends_strong_baseline_failure_to_experiment_plan(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["strong_baseline_gate"] = "failed"
    state.notes["strong_baseline_issues"] = "missing executed validation evidence"

    plan = build_rework_plan(state)

    assert plan is not None
    assert plan.rerun_from_phase == WorkflowPhase.EXPERIMENT_PLAN
    assert "experiment_report" in plan.refresh_artifacts
    assert plan.can_auto_apply is True


def test_rework_router_sends_innovation_failure_to_experiment_plan(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["innovation_evidence_gate"] = "failed"
    state.notes["innovation_evidence_issues"] = "stacking_ensemble: unsupported innovation claim"

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.EXPERIMENT_PLAN
    assert route.severity == "high"


def test_rework_router_sends_task_traceability_gaps_to_root_phase(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["task_traceability_gate"] = "failed"
    state.notes["task_traceability_blocking_issues"] = "Q1: missing executable model binding"
    assert recommend_rework_route(state).target_phase == WorkflowPhase.MODEL_DECISION

    state.notes["task_traceability_blocking_issues"] = "Q1: missing result table binding"
    assert recommend_rework_route(state).target_phase == WorkflowPhase.EVIDENCE_MAPPING

    state.notes["task_traceability_blocking_issues"] = "Q1: missing paper section binding"
    assert recommend_rework_route(state).target_phase == WorkflowPhase.SECTION_WRITING


def test_rework_router_sends_export_quality_failure_to_section_writing(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["export_quality_gate"] = "failed"
    state.notes["export_blocking_issues"] = "Submission blocker phrases remain in paper"

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.SECTION_WRITING
    assert route.severity == "high"


def test_rework_router_sends_award_structure_failure_to_section_writing(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["export_quality_gate"] = "failed"
    state.notes["export_blocking_issues"] = (
        "Award structure weak: missing high-value sections: validation, sensitivity"
    )

    route = recommend_rework_route(state)
    plan = build_rework_plan(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.SECTION_WRITING
    assert plan is not None
    assert any("national-contest section skeleton" in hint for hint in plan.repair_hints)


def test_rework_router_sends_missing_high_level_model_table_to_code_plan(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["export_quality_gate"] = "failed"
    state.notes["export_blocking_issues"] = (
        "Claimed high-level model has no matching generated result table: cvar_optimization"
    )

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.CODE_PLAN
    assert route.severity == "high"


def test_rework_router_sends_weak_risk_model_evidence_to_section_writing(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["export_quality_gate"] = "failed"
    state.notes["export_blocking_issues"] = (
        "Risk model evidence weak: cvar_optimization is claimed without at least 2 model-specific metrics."
    )

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.SECTION_WRITING
    assert route.severity == "high"


def test_rework_router_sends_paper_evidence_gate_failure_to_specific_phase(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["paper_evidence_gate"] = "failed"
    state.notes["paper_evidence_issues"] = (
        "Claimed high-level model has no matching generated result table: cvar_optimization"
    )

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.CODE_PLAN


def test_rework_router_sends_missing_selected_high_level_model_narrative_to_section_writing(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["paper_evidence_gate"] = "failed"
    state.notes["paper_evidence_issues"] = (
        "Selected high-level model missing from paper narrative: cvar_optimization"
    )

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.SECTION_WRITING


def test_rework_router_sends_high_level_table_metric_gap_to_code_plan(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["paper_evidence_gate"] = "failed"
    state.notes["paper_evidence_issues"] = (
        "High-level model table lacks model-specific metrics: cvar_optimization"
    )

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.CODE_PLAN

    plan = build_rework_plan(state)

    assert plan is not None
    assert any("cvar_optimization" in hint for hint in plan.repair_hints)


def test_rework_router_sends_missing_core_result_table_based_on_available_tables(temp_workspace):
    state = WorkflowState(problem_text="test", data_files=[], workspace=temp_workspace)
    state.notes["export_quality_gate"] = "failed"
    state.notes["export_blocking_issues"] = "Core result table missing: result section has no Markdown result table."

    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.RESULT_ANALYSIS

    (temp_workspace.tables_dir / "result_cvar_optimization.csv").write_text("cvar_loss\n45\n", encoding="utf-8")
    route = recommend_rework_route(state)

    assert route is not None
    assert route.target_phase == WorkflowPhase.SECTION_WRITING


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
    assert report.screenshot_manifest == [report.pages[0].screenshot]
    assert report.pages[0].content_bbox["x1"] > report.pages[0].content_bbox["x0"]
    assert report.pages[0].content_margin_px > 0


def test_pdf_render_layout_check_flags_edge_contact(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "edge_contact.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(1, A4[1] - 12, "text touches the edge")
    c.rect(0, 0, 30, A4[1], fill=1, stroke=0)
    c.save()

    report = check_pdf_render_layout(pdf, tmp_path / "edge_screenshots")

    assert report.pages_checked == 1
    assert report.passed is False
    assert any("boundary" in warning for warning in report.warnings)


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


def test_nipt_bmi_grouping_outputs_group_timing_strategy():
    df = pd.DataFrame(
        {
            "bmi": [23.5, 25.0, 27.8, 29.2, 30.5, 32.1, 34.0, 35.5],
            "gestational_week": [11, 12, 12, 13, 13, 14, 15, 15],
            "y_concentration": [0.032, 0.041, 0.044, 0.038, 0.046, 0.039, 0.043, 0.050],
            "abnormal": [0, 0, 0, 0, 1, 0, 1, 0],
        }
    )

    result = nipt_bmi_grouping(df)

    assert not result.empty
    assert {"bmi_group", "recommended_week", "threshold_reach_rate", "risk_level"}.issubset(result.columns)
    assert (result["recommended_week"] >= 12).all()


def test_model_selection_prefers_nipt_bmi_grouping_for_nipt_case():
    problem = "NIPT BMI gestational week Y chromosome concentration fetal abnormal risk grouping"

    result = ModelSelectionCrew().run(problem, [], [])

    selected = [item.model_id for item in result.selected]
    assert selected
    assert selected[0] == "nipt_bmi_grouping"


def test_crop_planting_plan_outputs_area_profit_strategy():
    df = pd.DataFrame(
        {
            "crop": ["wheat", "corn", "soybean", "rice"],
            "plot": ["A", "B", "C", "D"],
            "area": [30, 25, 20, 18],
            "yield_per_area": [1.2, 1.5, 0.9, 1.8],
            "price": [2.4, 2.1, 3.0, 2.8],
            "cost": [1.1, 1.3, 0.9, 1.7],
            "demand": [40, 35, 18, 32],
        }
    )

    result = crop_planting_plan(df)

    assert not result.empty
    assert {"allocated_area", "expected_profit", "demand_satisfaction_rate"}.issubset(result.columns)
    assert (result["allocated_area"] >= 0).all()


def test_model_selection_prefers_crop_planting_for_planting_case():
    problem = "crop planting farmland acreage yield price cost demand optimization strategy"

    result = ModelSelectionCrew().run(problem, [], [])

    selected = [item.model_id for item in result.selected]
    assert selected
    assert selected[0] == "crop_planting_plan"


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


def test_real_case_regression_supports_redacted_hidden_gold(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "hidden.txt").write_text(
        "crop planting farmland acreage yield price cost demand optimization strategy",
        encoding="utf-8",
    )
    corpus_index = tmp_path / "corpus.json"
    public_gold = tmp_path / "gold.json"
    hidden_gold = tmp_path / "hidden_gold.json"
    corpus_index.write_text(
        json.dumps(
            [
                {
                    "case_id": "hidden-case-a",
                    "year": 2024,
                    "problem": "C",
                    "title": "hidden crop planting",
                    "statement_path": "hidden.txt",
                    "attachment_paths": [],
                    "statement_chars": 70,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    public_gold.write_text("[]", encoding="utf-8")
    hidden_gold.write_text(
        json.dumps(
            [
                {
                    "case_id": "hidden-case-a",
                    "expected_task_types": ["optimization"],
                    "acceptable_primary_models": ["crop_planting_plan"],
                    "expected_numeric_ranges": [{"label": "profit", "min": 10, "max": 100}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_real_case_regression(
        corpus_index_path=corpus_index,
        gold_path=public_gold,
        hidden_gold_path=hidden_gold,
        corpus_root=corpus_root,
        output_dir=tmp_path / "out",
        min_average_score=60.0,
        min_primary_accuracy=0.0,
        min_candidate_coverage=1.0,
    )

    assert summary["hidden_gold_enabled"] is True
    assert summary["hidden_case_count"] == 1
    assert summary["answer_expectation_count"] == 1
    assert summary["scores"][0]["hidden"] is True
    assert summary["scores"][0]["case_id"].startswith("hidden:")
    assert summary["scores"][0]["selected_models"] == []


def test_real_case_regression_runner_executes_tiny_nipt_case(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    statement = corpus_root / "case-nipt.txt"
    statement.write_text(
        "NIPT BMI gestational week Y chromosome concentration fetal abnormal risk grouping.",
        encoding="utf-8",
    )
    corpus_index = tmp_path / "corpus.json"
    gold = tmp_path / "gold.json"
    corpus_index.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-nipt",
                    "year": 2025,
                    "problem": "C",
                    "title": "nipt bmi grouping",
                    "statement_path": "case-nipt.txt",
                    "attachment_paths": [],
                    "statement_chars": 75,
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
                    "case_id": "case-nipt",
                    "expected_task_types": ["statistics"],
                    "acceptable_primary_models": ["nipt_bmi_grouping"],
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
        min_primary_accuracy=0.0,
        min_candidate_coverage=1.0,
    )

    assert summary["schema_version"] == "1.1"
    assert summary["case_count"] == 1
    assert summary["scores"][0]["candidate_hit"] is True


def test_real_case_regression_runner_executes_tiny_planting_case(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    statement = corpus_root / "case-planting.txt"
    statement.write_text(
        "crop planting farmland acreage yield price cost demand optimization strategy",
        encoding="utf-8",
    )
    corpus_index = tmp_path / "corpus.json"
    gold = tmp_path / "gold.json"
    corpus_index.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-planting",
                    "year": 2024,
                    "problem": "C",
                    "title": "crop planting",
                    "statement_path": "case-planting.txt",
                    "attachment_paths": [],
                    "statement_chars": 70,
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
                    "case_id": "case-planting",
                    "expected_task_types": ["optimization"],
                    "acceptable_primary_models": ["crop_planting_plan"],
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
        min_primary_accuracy=0.0,
        min_candidate_coverage=1.0,
    )

    assert summary["case_count"] == 1
    assert summary["scores"][0]["candidate_hit"] is True
