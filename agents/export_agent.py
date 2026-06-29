from __future__ import annotations

from pathlib import Path

from agents.base import (
    A_INNOVATION_EVIDENCE_REPORT,
    A_EXPERIMENT_REPORT,
    A_PAPER,
    A_PAPER_QUALITY,
    A_PAPER_PDF_LAYOUT_REPORT,
    A_TASK_TRACEABILITY_REPORT,
    Agent,
    K_EXPORT_BLOCKING_ISSUES,
    K_EXPORT_ERRORS,
    K_EXPORT_PDF_LAYOUT_GATE,
    K_EXPORT_QUALITY_GATE,
    K_INNOVATION_EVIDENCE_GATE,
    K_INNOVATION_EVIDENCE_ISSUES,
    K_PAPER_EVIDENCE_SCORE,
    K_PAPER_EXPORT_SCORE,
    K_PAPER_QUALITY_REPORT,
    K_PAPER_QUALITY_SCORE,
    K_PAPER_SOLUTION_SCORE,
    K_PAPER_STRUCTURE_SCORE,
    K_STRONG_BASELINE_GATE,
    K_STRONG_BASELINE_ISSUES,
    K_TASK_TRACEABILITY_BLOCKING_ISSUES,
    K_TASK_TRACEABILITY_COVERAGE_PCT,
    K_TASK_TRACEABILITY_GATE,
    WorkflowState,
)
from models.catalog import EXECUTABLE_MODEL_LABELS
from tools.exporters import SUPPORTED_FORMATS, check_export_layout, export_document
from tools.file_tool import write_text
from tools.innovation_evidence import (
    build_innovation_evidence_report,
    innovation_evidence_blocking_issues,
    write_innovation_evidence_report,
)
from tools.paper_quality import (
    evaluate_paper_quality,
    format_quality_report,
    submission_blocking_issues,
)
from tools.pdf_layout_check import check_pdf_render_layout, write_pdf_layout_report
from tools.report_builder import build_document_from_paper
from tools.task_traceability import (
    build_task_traceability_report,
    task_traceability_blocking_issues,
    write_task_traceability_report,
)

_EXTRA_LABELS = {
    "capacity_gap": "需求容量缺口分析",
    "community_detection": "社群发现（规模/内部密度/核心成员）",
    "top5_communities": "内部连接密度最大的 5 个社群",
    "community_relation": "5 大社群间关系强度与重叠",
    "friend_recommendation": "好友推荐候选（链路预测得分）",
    "recommendation_reason": "Top-3 好友推荐及未成好友原因",
    "network_properties": "好友网络整体结构指标",
    "key_user_candidates": "关键用户候选（中心性+传播影响力）",
    "key_user_summary": "关键用户与 48 小时传播范围",
    "propagation_curve": "关键用户 48 小时传播曲线",
    "push_schedule": "推送名额优化方案",
    "push_strategy_comparison": "推送策略传播范围对比",
}


def export_paper(
    workspace,
    formats: list[str],
    title: str = "数学建模论文",
) -> dict[str, Path]:
    """Export the generated paper draft to the requested formats.

    Returns a mapping of format -> output path for the successful exports.
    """

    paper_path = workspace.paper_dir / "paper_draft.md"
    run_summary_path = workspace.logs_dir / "run_summary.json"
    labels = {**EXECUTABLE_MODEL_LABELS, **_EXTRA_LABELS}
    document = build_document_from_paper(paper_path, run_summary_path, labels, title=title)

    results: dict[str, Path] = {}
    errors: dict[str, str] = {}
    layout_warnings = check_export_layout(document)
    for fmt in formats:
        try:
            results[fmt] = export_document(document, fmt, workspace.paper_dir)
        except Exception as exc:  # keep other formats working if one backend is missing
            errors[fmt] = str(exc)
    if errors:
        results["_errors"] = errors  # type: ignore[assignment]
    if layout_warnings:
        results["_layout_warnings"] = layout_warnings  # type: ignore[assignment]
    pdf_path = results.get("pdf")
    if pdf_path:
        report = check_pdf_render_layout(
            pdf_path,
            workspace.paper_dir / "pdf_screenshots",
        )
        write_pdf_layout_report(report, workspace.paper_dir / "pdf_layout_report.json")
        if not report.passed:
            results["_pdf_layout_warnings"] = report.warnings  # type: ignore[assignment]
    return results


class ExportAgent(Agent):
    name = "export_agent"

    def __init__(self, formats: list[str] | None = None, title: str = "数学建模论文") -> None:
        self.formats = formats or list(SUPPORTED_FORMATS)
        self.title = title

    def run(self, state: WorkflowState) -> WorkflowState:
        if state.notes.get("traceability_gate") == "failed":
            state.notes[K_EXPORT_ERRORS] = (
                "Traceability gate failed; resolve unmapped numerical claims before export."
            )
            return state
        paper_path = state.artifacts.get(A_PAPER) or state.workspace.paper_dir / "paper_draft.md"
        if not Path(paper_path).exists():
            state.notes[K_EXPORT_ERRORS] = "Paper draft is missing; cannot export formal document."
            state.notes[K_EXPORT_QUALITY_GATE] = "failed"
            return state
        paper = Path(paper_path).read_text(encoding="utf-8")
        quality = evaluate_paper_quality(
            paper,
            workspace_root=state.workspace.root,
            available_figures=[p.name for p in state.workspace.figures_dir.glob("*.png")],
        )
        quality_report = format_quality_report(quality)
        quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", quality_report)
        state.artifacts[A_PAPER_QUALITY] = quality_path
        state.notes[K_PAPER_QUALITY_SCORE] = str(quality.score)
        state.notes[K_PAPER_QUALITY_REPORT] = quality_report
        state.notes[K_PAPER_SOLUTION_SCORE] = str(quality.metrics.get("solution_score", quality.score))
        state.notes[K_PAPER_EVIDENCE_SCORE] = str(quality.metrics.get("evidence_score", quality.score))
        state.notes[K_PAPER_STRUCTURE_SCORE] = str(quality.metrics.get("structure_score", quality.score))
        state.notes[K_PAPER_EXPORT_SCORE] = str(quality.metrics.get("export_score", quality.score))
        blockers = submission_blocking_issues(quality)
        task_traceability = build_task_traceability_report(
            deliverables=state.task_deliverable_specs,
            formulation=state.formulation_spec,
            registry=state.result_registry,
            paper_text=paper,
        )
        state.artifacts[A_TASK_TRACEABILITY_REPORT] = write_task_traceability_report(
            state.workspace,
            task_traceability,
        )
        state.notes[K_TASK_TRACEABILITY_GATE] = "passed" if task_traceability["passed"] else "failed"
        state.notes[K_TASK_TRACEABILITY_COVERAGE_PCT] = str(task_traceability["coverage_pct"])
        task_blockers = task_traceability_blocking_issues(task_traceability)
        if task_blockers:
            state.notes[K_TASK_TRACEABILITY_BLOCKING_ISSUES] = "; ".join(task_blockers)
            blockers.extend(f"Task traceability: {issue}" for issue in task_blockers)
        baseline_blockers = self._strong_baseline_blockers(state)
        if baseline_blockers:
            state.notes[K_STRONG_BASELINE_GATE] = "failed"
            state.notes[K_STRONG_BASELINE_ISSUES] = "; ".join(baseline_blockers)
            blockers.extend(f"Strong baseline: {issue}" for issue in baseline_blockers)
        innovation_report = build_innovation_evidence_report(
            state.workspace,
            paper_text=paper,
            model_selection_report=state.artifacts.get("model_selection_report"),
            experiment_report=state.artifacts.get(A_EXPERIMENT_REPORT),
        )
        state.artifacts[A_INNOVATION_EVIDENCE_REPORT] = write_innovation_evidence_report(
            state.workspace,
            innovation_report,
        )
        state.notes[K_INNOVATION_EVIDENCE_GATE] = (
            "passed" if innovation_report["passed"] else "failed"
        )
        innovation_blockers = innovation_evidence_blocking_issues(innovation_report)
        if innovation_blockers:
            state.notes[K_INNOVATION_EVIDENCE_ISSUES] = "; ".join(innovation_blockers)
            blockers.extend(f"Innovation evidence: {issue}" for issue in innovation_blockers)
        if blockers:
            state.notes[K_EXPORT_QUALITY_GATE] = "failed"
            state.notes[K_EXPORT_BLOCKING_ISSUES] = "; ".join(blockers)
            state.notes[K_EXPORT_ERRORS] = (
                "Paper quality gate failed; fix blocking issues before formal export."
            )
            return state
        state.notes[K_EXPORT_QUALITY_GATE] = "passed"
        results = export_paper(state.workspace, self.formats, title=self.title)
        errors = results.pop("_errors", None)  # type: ignore[assignment]
        layout_warnings = results.pop("_layout_warnings", None)  # type: ignore[assignment]
        pdf_layout_warnings = results.pop("_pdf_layout_warnings", None)  # type: ignore[assignment]
        for fmt, path in results.items():
            state.artifacts[f"paper_{fmt}"] = path
        pdf_layout_report = state.workspace.paper_dir / "pdf_layout_report.json"
        if pdf_layout_report.exists():
            state.artifacts[A_PAPER_PDF_LAYOUT_REPORT] = pdf_layout_report
        state.notes["export_formats"] = ", ".join(results.keys()) or "无"
        if layout_warnings:
            state.notes["export_layout_warnings"] = "; ".join(str(msg) for msg in layout_warnings)
        if pdf_layout_warnings:
            state.notes[K_EXPORT_PDF_LAYOUT_GATE] = "failed"
            state.notes["export_pdf_layout_warnings"] = "; ".join(str(msg) for msg in pdf_layout_warnings)
        elif pdf_layout_report:
            state.notes[K_EXPORT_PDF_LAYOUT_GATE] = "passed"
        if errors:
            state.notes[K_EXPORT_ERRORS] = "; ".join(f"{fmt}: {msg}" for fmt, msg in errors.items())
        return state

    def _strong_baseline_blockers(self, state: WorkflowState) -> list[str]:
        report_path = state.artifacts.get(A_EXPERIMENT_REPORT) or (
            state.workspace.logs_dir / "experiment_report.json"
        )
        if Path(report_path).exists():
            try:
                import json

                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return ["experiment report exists but cannot be parsed"]
            audit = report.get("strong_baseline_audit")
            if isinstance(audit, dict):
                if audit.get("passed") is True:
                    state.notes[K_STRONG_BASELINE_GATE] = "passed"
                    return []
                return [str(item) for item in audit.get("issues", []) if str(item)]
        if state.model_decision and state.model_decision.selected_model_ids:
            return ["missing experiment report with strong baseline and ablation audit"]
        return []
