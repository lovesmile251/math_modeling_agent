from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agents.base import K_EXECUTION_STATUS, K_RESULT_ANALYSIS, K_SELECTED_MODEL_IDS, WorkflowState
from tools.file_tool import write_text
from tools.model_ids import normalize_model_ids


OPTIMIZATION_MODEL_TOKENS = (
    "optimization",
    "allocation",
    "knapsack",
    "assignment",
    "packing",
    "scheduling",
    "route",
    "control",
    "portfolio",
    "pricing",
    "esp",
)


@dataclass(frozen=True)
class PreWritingGateReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, int | str] = field(default_factory=dict)


def evaluate_pre_writing_gate(state: WorkflowState) -> PreWritingGateReport:
    """Verify that writing has enough real evidence to avoid fabricated papers."""

    issues: list[str] = []
    suggestions: list[str] = []
    table_files = sorted(state.workspace.tables_dir.glob("*.csv"))
    figure_files = sorted(state.workspace.figures_dir.glob("*.png"))
    selected_models = _selected_models(state)
    model_output_count = _count_model_outputs(state.workspace.logs_dir / "run_summary.json")
    has_execution_context = (
        K_EXECUTION_STATUS in state.notes
        or bool(state.data_files)
        or (state.workspace.logs_dir / "run_summary.json").exists()
        or bool(table_files)
    )
    if not has_execution_context:
        return PreWritingGateReport(
            ok=True,
            metrics={
                "tables": len(table_files),
                "figures": len(figure_files),
                "model_outputs": model_output_count,
                "selected_models": len(selected_models),
                "mode": "ad_hoc_writing",
            },
        )

    if state.notes.get(K_EXECUTION_STATUS) != "success":
        issues.append("代码执行未成功，不能进入最终论文写作。")
        suggestions.append("先修复代码执行错误，确保结果表和图表真实生成。")

    if not table_files:
        issues.append("未生成任何结果表，论文缺少可追溯数值证据。")
        suggestions.append("至少产出描述统计表和一个核心模型结果表后再写作。")

    if state.notes.get("formulation_status") == "needs_revision":
        issues.append("数学建模公式规格未通过校验：" + state.notes.get("formulation_issues", "未知问题"))
        suggestions.append("先修复模型 ID、任务类型或变量/目标函数缺口。")

    result_analysis = state.notes.get(K_RESULT_ANALYSIS, "")
    if len(result_analysis.strip()) < 80:
        issues.append("结果分析不足，尚未形成可写入论文的结论链。")
        suggestions.append("补充每张核心结果表的指标解释、约束满足情况和对题目问题的回答。")

    if _needs_optimization_result(state.problem_text, selected_models):
        optimization_tables = [
            path for path in table_files
            if any(token in path.stem.lower() for token in OPTIMIZATION_MODEL_TOKENS)
        ]
        if not optimization_tables:
            issues.append("题目或模型选择包含优化任务，但未发现优化结果表。")
            suggestions.append("先运行优化求解器，输出目标函数值、决策变量和约束满足情况。")

    metrics: dict[str, int | str] = {
        "tables": len(table_files),
        "figures": len(figure_files),
        "model_outputs": model_output_count,
        "selected_models": len(selected_models),
    }
    return PreWritingGateReport(ok=not issues, issues=issues, suggestions=suggestions, metrics=metrics)


def format_pre_writing_gate_report(report: PreWritingGateReport) -> str:
    status = "通过" if report.ok else "阻断"
    lines = [
        "## 写作前证据门禁",
        f"- 状态：{status}",
        f"- 结果表数量：{report.metrics.get('tables', 0)}",
        f"- 图表数量：{report.metrics.get('figures', 0)}",
        f"- 模型结果数量：{report.metrics.get('model_outputs', 0)}",
        f"- 入选模型数量：{report.metrics.get('selected_models', 0)}",
        "",
        "### 问题",
    ]
    lines.extend(f"- {item}" for item in (report.issues or ["未发现阻断项。"]))
    lines.append("")
    lines.append("### 建议")
    lines.extend(f"- {item}" for item in (report.suggestions or ["可以进入论文写作。"]))
    return "\n".join(lines)


def write_pre_writing_gate_report(state: WorkflowState, report: PreWritingGateReport) -> Path:
    return write_text(
        state.workspace.logs_dir / "prewriting_gate_report.md",
        format_pre_writing_gate_report(report),
    )


def _selected_models(state: WorkflowState) -> list[str]:
    raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        parsed = []
    return normalize_model_ids(parsed).selected


def _needs_optimization_result(problem_text: str, selected_models: list[str]) -> bool:
    text = problem_text.lower()
    if any(token in model_id for model_id in selected_models for token in OPTIMIZATION_MODEL_TOKENS):
        return True
    return any(term in text for term in ("优化", "最优", "最小", "最大", "降低", "控制", "optimization", "optimal"))


def _count_model_outputs(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, list):
        return 0
    count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        outputs = item.get("model_outputs") or {}
        if isinstance(outputs, dict):
            count += len(outputs)
        runs = item.get("model_runs") or []
        if isinstance(runs, list):
            count += sum(
                1
                for run in runs
                if isinstance(run, dict)
                and run.get("status") == "success"
                and run.get("table")
            )
    return count
