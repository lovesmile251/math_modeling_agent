from __future__ import annotations

import json

import workflows.modeling_workflow as wf
from agents.base import (
    A_PAPER,
    A_PAPER_DOCX_LAYOUT_REPORT,
    A_PAPER_EVIDENCE_AUDIT,
    FormulationSpec,
    ModelDecision,
    ResultRegistry,
    TaskDeliverableSpec,
    WorkflowState,
)
from agents.export_agent import ExportAgent, export_paper
from agents.base import WorkflowPhase

PROBLEM = "请根据附件数据，预测需求的未来趋势，并对容量缺口进行综合评价。"


def test_workflow_end_to_end(project_rooted_workspace, sample_dataframe):
    (project_rooted_workspace.data_dir / "sample.csv").write_text(
        sample_dataframe.to_csv(index=False), encoding="utf-8"
    )

    state = wf.ModelingWorkflow(use_llm=False, workspace=project_rooted_workspace).run(PROBLEM)

    # Core artifacts must be produced.
    assert state.notes.get("execution_status") == "success"
    assert state.artifacts["code"].exists()
    assert state.artifacts["paper"].exists()
    assert state.artifacts["review"].exists()
    assert state.artifacts["paper_quality"].exists()

    # The run summary and at least one table/figure should be generated.
    assert (project_rooted_workspace.logs_dir / "run_summary.json").exists()
    assert list(project_rooted_workspace.tables_dir.glob("*.csv"))
    assert list(project_rooted_workspace.figures_dir.glob("*.png"))

    paper_text = state.artifacts["paper"].read_text(encoding="utf-8")
    assert "问题重述" in paper_text
    assert "国奖质量门禁" in state.artifacts["paper_quality"].read_text(encoding="utf-8")


def test_workflow_with_export(project_rooted_workspace, sample_dataframe):
    (project_rooted_workspace.data_dir / "sample.csv").write_text(
        sample_dataframe.to_csv(index=False), encoding="utf-8"
    )

    state = wf.ModelingWorkflow(
        use_llm=False,
        export_formats=["docx", "pdf", "latex"],
        workspace=project_rooted_workspace,
    ).run(PROBLEM)

    assert state.notes["export_quality_gate"] == "failed"
    assert "Paper quality gate failed" in state.notes["export_errors"]
    assert state.notes["export_blocking_issues"]

    for fmt in ("docx", "pdf", "latex"):
        assert f"paper_{fmt}" not in state.artifacts


def test_export_paper_standalone(project_rooted_workspace):
    # A minimal paper draft is enough for the exporter to produce all formats.
    (project_rooted_workspace.paper_dir / "paper_draft.md").write_text(
        "# 论文\n\n## 摘要\n本文测试导出。\n", encoding="utf-8"
    )
    results = export_paper(project_rooted_workspace, ["docx", "pdf", "latex"])
    results.pop("_errors", None)
    assert set(results.keys()) == {"docx", "pdf", "latex"}
    for path in results.values():
        assert path.exists()
    docx_layout_report = project_rooted_workspace.paper_dir / "docx_template_layout_report.json"
    assert docx_layout_report.exists()
    payload = json.loads(docx_layout_report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert any(item["role"] == "title" for item in payload["field_mapping"])


def test_export_agent_blocks_formal_export_for_submission_blockers(project_rooted_workspace):
    paper_path = project_rooted_workspace.paper_dir / "paper_draft.md"
    paper_path.write_text(
        "# 测试论文\n\n## 摘要\n本文后续补充结果。\n\n## 结果分析\n暂无。\n",
        encoding="utf-8",
    )
    state = WorkflowState(
        problem_text="test",
        data_files=[],
        workspace=project_rooted_workspace,
    )
    state.artifacts[A_PAPER] = paper_path

    state = ExportAgent(formats=["docx", "pdf", "latex"]).run(state)

    assert state.notes["export_quality_gate"] == "failed"
    assert "Paper quality gate failed" in state.notes["export_errors"]
    assert "Submission blocker" in state.notes["export_blocking_issues"]
    assert state.notes["paper_evidence_gate"] == "failed"
    assert A_PAPER_EVIDENCE_AUDIT in state.artifacts
    assert state.artifacts[A_PAPER_EVIDENCE_AUDIT].exists()
    assert (project_rooted_workspace.logs_dir / "paper_evidence_audit.md").exists()
    assert A_PAPER_DOCX_LAYOUT_REPORT not in state.artifacts
    assert "paper_docx" not in state.artifacts
    assert "paper_pdf" not in state.artifacts
    assert "paper_latex" not in state.artifacts


def test_export_agent_legacy_claim_traceability_does_not_skip_formal_gates(project_rooted_workspace):
    paper_path = project_rooted_workspace.paper_dir / "paper_draft.md"
    paper_path.write_text(
        "# Paper\n\n## Abstract\nThis draft has one unsupported result.\n\n## Results\nNo table yet.\n",
        encoding="utf-8",
    )
    state = WorkflowState(problem_text="test", data_files=[], workspace=project_rooted_workspace)
    state.artifacts[A_PAPER] = paper_path
    state.notes["traceability_gate"] = "failed"

    state = ExportAgent(formats=["docx"]).run(state)

    assert state.notes["export_quality_gate"] == "failed"
    assert state.notes["paper_evidence_gate"] == "failed"
    assert state.artifacts[A_PAPER_EVIDENCE_AUDIT].exists()
    assert "Traceability gate failed" not in state.notes["export_errors"]
    assert "paper_docx" not in state.artifacts


def test_export_agent_blocks_missing_task_paper_binding(project_rooted_workspace):
    (project_rooted_workspace.tables_dir / "resource_allocation_plan.csv").write_text(
        "decision,value\nx,1\n",
        encoding="utf-8",
    )
    paper_path = project_rooted_workspace.paper_dir / "paper_draft.md"
    paper_path.write_text(
        """# 测试论文

## 摘要
本文建立优化模型并完成结果验证，核心指标为 1，正文引用文献 [1]。

## 关键词
优化模型；结果验证；敏感性分析

## 问题重述
需要求解资源配置方案。

## 问题分析
本文分析数据约束和目标函数。

## 模型假设
假设数据可靠。

## 符号说明
| 符号 | 含义 |
| --- | --- |
| x | 决策变量 |

## 模型建立
\\(x=1\\)

## 结果分析
| 指标 | 数值 |
| --- | --- |
| 目标值 | 1 |

## 模型检验与误差分析
进行误差、敏感性、检验、对比和基准分析。

## 模型评价与推广
模型可推广到同类资源配置问题。

## 参考文献
[1] Zhang. Model validation. Journal, 2024.

## 附录
代码见附录。
""",
        encoding="utf-8",
    )
    state = WorkflowState(problem_text="test", data_files=[], workspace=project_rooted_workspace)
    state.artifacts[A_PAPER] = paper_path
    state.task_deliverable_specs = [
        TaskDeliverableSpec(
            task_id="Q2",
            task_type="optimization",
            objective="给出最优方案",
            required_tables=["optimization"],
        )
    ]
    state.formulation_spec = FormulationSpec(
        stages=[{"stage_id": "Q2", "task_type": "optimization", "model_ids": ["resource_allocation"]}]
    )
    state.result_registry = ResultRegistry(
        entries=[
            {
                "type": "table",
                "name": "resource_allocation_plan",
                "path": str(project_rooted_workspace.tables_dir / "resource_allocation_plan.csv"),
                "model_id": "resource_allocation",
            }
        ]
    )

    state = ExportAgent(formats=["docx"]).run(state)

    assert state.notes["task_traceability_gate"] == "failed"
    assert "Q2: missing paper section binding" in state.notes["task_traceability_blocking_issues"]
    assert "paper_docx" not in state.artifacts


def test_export_agent_blocks_missing_strong_baseline_audit(project_rooted_workspace):
    paper_path = project_rooted_workspace.paper_dir / "paper_draft.md"
    paper_path.write_text(
        """# 测试论文

## 摘要
本文完成模型计算，核心目标值为 1，并引用文献 [1]。

## 关键词
预测模型；基线对比；消融实验

## 问题重述
问题要求完成预测。

## 问题分析
本文对预测任务进行分析。

## 模型假设
假设数据可靠。

## 符号说明
| 符号 | 含义 |
| --- | --- |
| y | 目标变量 |

## 模型建立
\\(y=1\\)

## 结果分析
| 指标 | 数值 |
| --- | --- |
| 目标值 | 1 |

## 模型检验与误差分析
进行误差、敏感性、检验、对比和基准分析。

## 模型评价与推广
模型可推广。

## 参考文献
[1] Zhang. Model validation. Journal, 2024.

## 附录
代码见附录。
""",
        encoding="utf-8",
    )
    state = WorkflowState(problem_text="test", data_files=[], workspace=project_rooted_workspace)
    state.artifacts[A_PAPER] = paper_path
    state.model_decision = ModelDecision(
        primary_model_id="trend_forecast",
        baseline_model_id="smoothing_forecast",
        selected_model_ids=["trend_forecast", "smoothing_forecast"],
    )

    state = ExportAgent(formats=["docx"]).run(state)

    assert state.notes["strong_baseline_gate"] == "failed"
    assert "missing experiment report" in state.notes["strong_baseline_issues"]
    assert "paper_docx" not in state.artifacts


def test_export_agent_blocks_unsupported_innovation_claim(project_rooted_workspace):
    paper_path = project_rooted_workspace.paper_dir / "paper_draft.md"
    paper_path.write_text(
        """# Test Paper

## Abstract
This paper solves the task and reports a result value of 1 with citation [1].

## Keywords
forecasting; validation; sensitivity

## Model
Model innovation: we use a Stacking ensemble to improve prediction stability.

## Results
| metric | value |
| --- | --- |
| objective | 1 |

## Validation
We compare errors and discuss sensitivity.

## References
[1] Zhang. Model validation. Journal, 2024.
""",
        encoding="utf-8",
    )
    state = WorkflowState(problem_text="test", data_files=[], workspace=project_rooted_workspace)
    state.artifacts[A_PAPER] = paper_path

    state = ExportAgent(formats=["docx"]).run(state)

    assert state.notes["innovation_evidence_gate"] == "failed"
    assert "stacking_ensemble" in state.notes["innovation_evidence_issues"]
    assert "paper_docx" not in state.artifacts


def test_run_workspace_uses_isolated_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(wf, "PROJECT_ROOT", tmp_path)

    workflow = wf.ModelingWorkflow(use_llm=False, run_workspace=True)
    workflow.agents = []

    state = workflow.run("test problem", data_files=[])

    assert state.workspace.root.parent == tmp_path / "workspace" / "runs"
    assert state.workspace.root.name.startswith("20")
    assert state.workspace.effective_project_root == tmp_path
    assert (state.workspace.input_dir / "problem.txt").read_text(encoding="utf-8") == "test problem"

    diagnostics = json.loads(
        (state.workspace.logs_dir / "workflow_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["workspace_root"] == str(state.workspace.root)


def test_explicit_empty_data_files_disable_auto_discovery(project_rooted_workspace, sample_dataframe):
    (project_rooted_workspace.data_dir / "sample.csv").write_text(
        sample_dataframe.to_csv(index=False), encoding="utf-8"
    )
    workflow = wf.ModelingWorkflow(use_llm=False, workspace=project_rooted_workspace)
    workflow.agents = []

    state = workflow.run("statement-only problem", data_files=[])

    assert state.data_files == []


def test_workflow_auto_rework_reruns_from_recommended_phase(project_rooted_workspace):
    class FakeExportAgent:
        name = "export_agent"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, state):
            self.calls += 1
            if self.calls == 1:
                state.notes["export_quality_gate"] = "failed"
                state.notes["export_blocking_issues"] = "Submission blocker phrases remain in paper"
            else:
                state.notes["export_quality_gate"] = "passed"
                state.notes.pop("export_blocking_issues", None)
            return state

    fake_export = FakeExportAgent()
    workflow = wf.ModelingWorkflow(
        use_llm=False,
        workspace=project_rooted_workspace,
        auto_rework_attempts=1,
    )
    workflow.agents = [fake_export]
    workflow._invalidate_phase_cache()

    state = workflow.run("test problem")

    assert fake_export.calls == 2
    assert state.notes["auto_rework_status"] == "resolved"
    assert state.notes["auto_rework_rerun_from_phase"] == WorkflowPhase.SECTION_WRITING.value
    assert state.notes["auto_rework_applied"] == "true"
    assert state.artifacts["auto_rework_report"].exists()
    assert state.artifacts["auto_rework_report_md"].exists()
    assert state.artifacts["workflow_gate_summary"].exists()
    assert state.artifacts["workflow_gate_summary_md"].exists()

    report = json.loads(state.artifacts["auto_rework_report"].read_text(encoding="utf-8"))
    assert report["status"] == "resolved"
    assert report["initial_route"]["target_phase"] == WorkflowPhase.SECTION_WRITING.value
    assert report["repair_hints"]
    assert report["before_gates"]["export_quality_gate"] == "failed"
    assert report["after_gates"]["export_quality_gate"] == "passed"
    assert "自动返工报告" in state.artifacts["auto_rework_report_md"].read_text(encoding="utf-8")
    assert "Repair Hints" in state.artifacts["auto_rework_report_md"].read_text(encoding="utf-8")
    summary = json.loads(state.artifacts["workflow_gate_summary"].read_text(encoding="utf-8"))
    assert summary["gates"]["export_quality_gate"] == "passed"


def test_workflow_auto_rework_fuses_same_cause_repeat(project_rooted_workspace):
    class AlwaysFailingExportAgent:
        name = "export_agent"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, state):
            self.calls += 1
            state.notes["export_quality_gate"] = "failed"
            state.notes["export_blocking_issues"] = "Submission blocker phrases remain in paper"
            return state

    fake_export = AlwaysFailingExportAgent()
    workflow = wf.ModelingWorkflow(
        use_llm=False,
        workspace=project_rooted_workspace,
        auto_rework_attempts=3,
    )
    workflow.agents = [fake_export]
    workflow._invalidate_phase_cache()

    state = workflow.run("test problem")

    assert fake_export.calls == 2
    assert state.notes["auto_rework_status"] == "fused_same_cause"
    assert state.notes["auto_rework_fuse_signature"]
    report = json.loads(state.artifacts["auto_rework_report"].read_text(encoding="utf-8"))
    assert report["status"] == "fused_same_cause"
