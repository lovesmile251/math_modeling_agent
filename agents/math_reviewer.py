from __future__ import annotations

import logging
from pathlib import Path

from agents.base import (
    A_PAPER,
    Agent,
    ReviewFindings,
    WorkflowState,
)
from tools.prompt_loader import load_prompt

log = logging.getLogger("mma.math_reviewer")


class MathReviewerAgent(Agent):
    """Checks mathematical consistency: formulas, symbols, assumptions vs code.

    Verifies that every formula in the paper matches what the code actually
    implements, symbols are used consistently, and assumptions are stated.
    """

    name = "math_reviewer"

    def run(self, state: WorkflowState) -> WorkflowState:
        findings = ReviewFindings(reviewer="math", score=80)
        paper_text = self._read_paper(state)

        if state.llm and state.llm.enabled and paper_text:
            try:
                prompt = load_prompt("math_review.md")
                response = state.llm.complete(prompt, self._build_llm_input(state, paper_text))
                findings = self._parse_response(response, findings)
                state.notes["math_review_score"] = str(findings.score)
            except Exception as exc:
                state.notes["math_review_error"] = str(exc)
                log.warning("LLM math review failed: %s", exc)
                findings = self._heuristic_math_review(state, paper_text)
        else:
            findings = self._heuristic_math_review(state, paper_text)

        state.review_findings = findings
        state.notes["math_review_raw"] = findings.raw_report
        return state

    def _read_paper(self, state: WorkflowState) -> str:
        pp = state.artifacts.get(A_PAPER)
        if pp and pp.exists():
            return pp.read_text(encoding="utf-8", errors="replace")
        return ""

    def _build_llm_input(self, state: WorkflowState, paper_text: str) -> str:
        code_path = state.artifacts.get("code")
        code_text = ""
        if code_path and code_path.exists():
            code_text = code_path.read_text(encoding="utf-8", errors="replace")[:6000]

        return "\n\n".join([
            "=== 论文 ===",
            paper_text[:8000],
            "=== 生成代码 ===",
            code_text,
            "检查要点：",
            "1. 论文中的公式是否与代码中的实现一致",
            "2. 符号使用是否前后一致（同一变量不同符号？不同变量同一符号？）",
            "3. 模型假设是否在论文中明确列出",
            "4. 数学推导是否有逻辑跳跃或错误",
            "5. 算法步骤是否与代码执行顺序一致",
        ])

    def _parse_response(self, response: str, base: ReviewFindings) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("问题") or s.startswith("Issue") or "不一致" in s or "错误" in s:
                issues.append({"description": s.lstrip("-*# "), "severity": "warning", "category": "math"})
            elif s.startswith("建议") or s.startswith("Suggestion"):
                suggestions.append(s.lstrip("-*# "))
        score = 80 - min(len(issues) * 10, 50)
        return ReviewFindings(
            reviewer="math",
            score=max(score, 30),
            issues=issues or base.issues,
            suggestions=suggestions or base.suggestions,
            raw_report=response,
        )

    def _heuristic_math_review(self, state: WorkflowState, paper_text: str) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []

        # check for equation presence
        import re
        eq_count = len(re.findall(r"\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)", paper_text, re.DOTALL))
        if eq_count < 3:
            issues.append({"description": "论文化学公式少于3个，可能缺少数学建模内容", "severity": "warning", "category": "math"})
        if eq_count >= 3:
            suggestions.append(f"检测到 {eq_count} 个内联/独立公式，建议检查与代码的一致性")

        # check for symbol notation table
        if "符号" not in paper_text and "notation" not in paper_text.lower():
            issues.append({"description": "缺少符号说明表，建议添加以提升可读性", "severity": "info", "category": "math"})

        # check for assumption section
        if "假设" not in paper_text:
            issues.append({"description": "缺少模型假设章节，建议补充", "severity": "warning", "category": "math"})

        score = 80 - min(len([i for i in issues if i.get("severity") == "warning"]) * 15, 50)
        return ReviewFindings(
            reviewer="math", score=score, issues=issues, suggestions=suggestions,
            raw_report=f"启发式数学审查：{eq_count}个公式，{len(issues)}个问题",
        )
