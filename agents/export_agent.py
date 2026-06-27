from __future__ import annotations

from pathlib import Path

from agents.base import Agent, WorkflowState
from models.catalog import EXECUTABLE_MODEL_LABELS
from tools.exporters import SUPPORTED_FORMATS, check_export_layout, export_document
from tools.report_builder import build_document_from_paper

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
    return results


class ExportAgent(Agent):
    name = "export_agent"

    def __init__(self, formats: list[str] | None = None, title: str = "数学建模论文") -> None:
        self.formats = formats or list(SUPPORTED_FORMATS)
        self.title = title

    def run(self, state: WorkflowState) -> WorkflowState:
        if state.notes.get("traceability_gate") == "failed":
            state.notes["export_errors"] = (
                "Traceability gate failed; resolve unmapped numerical claims before export."
            )
            return state
        results = export_paper(state.workspace, self.formats, title=self.title)
        errors = results.pop("_errors", None)  # type: ignore[assignment]
        layout_warnings = results.pop("_layout_warnings", None)  # type: ignore[assignment]
        for fmt, path in results.items():
            state.artifacts[f"paper_{fmt}"] = path
        state.notes["export_formats"] = ", ".join(results.keys()) or "无"
        if layout_warnings:
            state.notes["export_layout_warnings"] = "; ".join(str(msg) for msg in layout_warnings)
        if errors:
            state.notes["export_errors"] = "; ".join(f"{fmt}: {msg}" for fmt, msg in errors.items())
        return state
