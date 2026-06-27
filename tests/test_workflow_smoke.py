from __future__ import annotations

import json

import workflows.modeling_workflow as wf
from agents.export_agent import export_paper

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

    for fmt in ("docx", "pdf", "latex"):
        key = f"paper_{fmt}"
        assert key in state.artifacts, f"缺少导出产物 {key}"
        assert state.artifacts[key].exists()


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
