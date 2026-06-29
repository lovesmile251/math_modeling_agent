from __future__ import annotations

from agents.base import FormulationSpec, ResultRegistry, TaskDeliverableSpec
from tools.task_traceability import build_task_traceability_report, task_traceability_blocking_issues


def test_task_traceability_requires_model_table_and_paper_section():
    deliverables = [
        TaskDeliverableSpec(
            task_id="Q1",
            task_type="forecast",
            objective="预测需求趋势",
            required_tables=["forecast"],
        )
    ]
    formulation = FormulationSpec(
        stages=[
            {
                "stage_id": "Q1",
                "task_type": "forecast",
                "model_ids": ["trend_forecast"],
            }
        ]
    )
    registry = ResultRegistry(
        entries=[
            {
                "type": "table",
                "name": "sample_trend_forecast",
                "path": "tables/sample_trend_forecast.csv",
                "model_id": "trend_forecast",
            }
        ]
    )
    paper = "# 论文\n\n## 问题一：预测需求趋势\n\n趋势预测结果见核心结果表。"

    report = build_task_traceability_report(
        deliverables=deliverables,
        formulation=formulation,
        registry=registry,
        paper_text=paper,
    )

    assert report["passed"] is True
    assert report["coverage_pct"] == 100.0
    assert report["items"][0]["model_ok"] is True
    assert report["items"][0]["table_ok"] is True
    assert report["items"][0]["paper_ok"] is True


def test_task_traceability_reports_missing_paper_section():
    deliverables = [
        TaskDeliverableSpec(
            task_id="Q2",
            task_type="optimization",
            objective="给出最优方案",
            required_tables=["optimization"],
        )
    ]
    formulation = FormulationSpec(
        stages=[
            {
                "stage_id": "Q2",
                "task_type": "optimization",
                "model_ids": ["resource_allocation"],
            }
        ]
    )
    registry = ResultRegistry(
        entries=[
            {
                "type": "table",
                "name": "resource_allocation_plan",
                "path": "tables/resource_allocation_plan.csv",
                "model_id": "resource_allocation",
            }
        ]
    )

    report = build_task_traceability_report(
        deliverables=deliverables,
        formulation=formulation,
        registry=registry,
        paper_text="# 论文\n\n## 问题一\n这里只讨论预测。",
    )

    assert report["passed"] is False
    assert any("missing paper section binding" in issue for issue in report["issues"])
    assert task_traceability_blocking_issues(report)
