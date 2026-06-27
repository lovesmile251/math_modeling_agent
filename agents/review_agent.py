from __future__ import annotations

import json
import logging

from agents.base import (
    A_CODE,
    A_PAPER,
    A_REVIEW,
    K_EXECUTION_ATTEMPTS,
    K_EXECUTION_STATUS,
    K_MODELING_PLAN,
    K_PAPER_EVIDENCE_SCORE,
    K_PAPER_EXPORT_SCORE,
    K_PAPER_QUALITY_REPORT,
    K_PAPER_QUALITY_SCORE,
    K_PAPER_SOLUTION_SCORE,
    K_PAPER_STRUCTURE_SCORE,
    K_RESULT_ANALYSIS,
    K_REVIEW_REPORT,
    Agent,
    WorkflowState,
)
from tools.file_tool import write_text
from tools.paper_quality import evaluate_paper_quality, format_quality_report

log = logging.getLogger("mma.review_agent")


class ReviewAgent(Agent):
    name = "review_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        findings: list[str] = []
        suggestions: list[str] = []

        if state.notes.get(K_EXECUTION_STATUS) != "success":
            findings.append("代码执行未成功，论文结果部分不能作为最终结论。")
            suggestions.append("优先修复执行错误，再生成最终论文。")

        if A_CODE not in state.artifacts:
            findings.append("缺少可运行代码文件。")

        table_files = sorted(state.workspace.tables_dir.glob("*"))
        figure_files = sorted(state.workspace.figures_dir.glob("*.png"))
        if state.data_files and not table_files:
            findings.append("提供了数据文件，但未生成结果表。")
        if state.data_files and not figure_files:
            findings.append("提供了数据文件，但未生成图表。")

        model_output_count = self._count_model_outputs(state)
        if state.data_files and model_output_count == 0:
            findings.append("没有任何模型成功产出结果表：论文不得编造模型结论，只能基于描述统计与题目展开。")
            suggestions.append("检查模型选择与数据字段是否匹配，确保至少一个模型产出真实结果后再下结论。")
        else:
            suggestions.append(
                f"论文中的每个数值、排名、参数都必须对应已产出的 {model_output_count} 张结果表，"
                "严禁使用 U_a、$U_{key}$、“假设为”等占位符。"
            )

        result_analysis = state.notes.get(K_RESULT_ANALYSIS, "")
        if len(result_analysis.strip()) < 80:
            findings.append("结果分析内容偏少。")
            suggestions.append("补充指标解释、图表解释和对题目问题的回答。")

        if not state.notes.get(K_MODELING_PLAN):
            findings.append("缺少建模方案。")

        paper_quality_text = ""
        paper_path = state.artifacts.get(A_PAPER)
        if paper_path and paper_path.exists():
            paper = paper_path.read_text(encoding="utf-8")
            quality = evaluate_paper_quality(
                paper,
                workspace_root=state.workspace.root,
                available_figures=[p.name for p in state.workspace.figures_dir.glob("*.png")],
            )
            state.notes[K_PAPER_QUALITY_SCORE] = str(quality.score)
            state.notes[K_PAPER_QUALITY_REPORT] = format_quality_report(quality)
            state.notes[K_PAPER_SOLUTION_SCORE] = str(quality.metrics.get("solution_score", quality.score))
            state.notes[K_PAPER_EVIDENCE_SCORE] = str(quality.metrics.get("evidence_score", quality.score))
            state.notes[K_PAPER_STRUCTURE_SCORE] = str(quality.metrics.get("structure_score", quality.score))
            state.notes[K_PAPER_EXPORT_SCORE] = str(quality.metrics.get("export_score", quality.score))
            paper_quality_text = format_quality_report(quality)
            if quality.score < 82:
                findings.append(f"论文未达到国奖质量门禁：当前质量分 {quality.score}/100。")
                suggestions.extend(quality.suggestions)
            gate_failures = [
                issue
                for issue in quality.issues
                if any(
                    marker in issue
                    for marker in (
                        "Submission blocker",
                        "Reference citations without matching entries",
                        "Reference entries not cited in body",
                        "Reference section is missing",
                        "Core result table missing",
                    )
                )
            ]
            if gate_failures:
                findings.extend(gate_failures)
                suggestions.extend(quality.suggestions)

        if not findings:
            findings.append("基础闭环完整：代码、执行日志、结果分析和论文素材均已生成。")

        if not suggestions:
            suggestions.append("下一步可补充灵敏度分析、误差分析和模型对比。")

        review = "\n".join(
            [
                "# 审稿检查",
                "",
                "## 发现",
                *(f"- {item}" for item in findings),
                "",
                "## 修改建议",
                *(f"- {item}" for item in suggestions),
                "",
                "## 已生成资产",
                f"- 结果表数量：{len(table_files)}",
                f"- 图表数量：{len(figure_files)}",
                f"- 成功产出结果的模型数：{model_output_count}",
                f"- 执行尝试次数：{state.notes.get(K_EXECUTION_ATTEMPTS, '0')}",
                "",
                paper_quality_text,
            ]
        )
        review_path = write_text(state.workspace.paper_dir / "review_report.md", review)
        state.artifacts[A_REVIEW] = review_path
        state.notes[K_REVIEW_REPORT] = review
        return state

    def _count_model_outputs(self, state: WorkflowState) -> int:
        summary_path = state.workspace.logs_dir / "run_summary.json"
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
            if isinstance(item, dict):
                model_outputs = item.get("model_outputs") or {}
                if isinstance(model_outputs, dict):
                    count += len(model_outputs)
                model_runs = item.get("model_runs") or []
                if isinstance(model_runs, list):
                    count += sum(
                        1
                        for run in model_runs
                        if isinstance(run, dict)
                        and run.get("status") == "success"
                        and run.get("table")
                    )
        return count
