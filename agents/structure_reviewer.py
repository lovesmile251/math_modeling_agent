from __future__ import annotations

import logging

from agents.base import (
    A_PAPER,
    Agent,
    ReviewFindings,
    WorkflowState,
)

log = logging.getLogger("mma.structure_reviewer")


class StructureReviewerAgent(Agent):
    """Checks argument completeness: every sub-question must form a closed loop
    (analysis → model → solution → result → interpretation).
    """

    name = "structure_reviewer"

    def run(self, state: WorkflowState) -> WorkflowState:
        findings = ReviewFindings(reviewer="structure", score=80)
        paper_text = self._read_paper(state)

        if state.llm and state.llm.enabled and paper_text:
            try:
                from tools.prompt_loader import load_prompt
                prompt = load_prompt("structure_review.md")
                response = state.llm.complete(prompt, self._build_llm_input(state, paper_text))
                findings = self._parse_response(response, findings)
                state.notes["structure_review_score"] = str(findings.score)
            except Exception as exc:
                state.notes["structure_review_error"] = str(exc)
                log.warning("LLM structure review failed: %s", exc)
                findings = self._heuristic_structure_review(state, paper_text)
        else:
            findings = self._heuristic_structure_review(state, paper_text)

        state.review_findings = findings
        state.notes["structure_review_raw"] = findings.raw_report
        return state

    def _read_paper(self, state: WorkflowState) -> str:
        pp = state.artifacts.get(A_PAPER)
        if pp and pp.exists():
            return pp.read_text(encoding="utf-8", errors="replace")
        return ""

    def _build_llm_input(self, state: WorkflowState, paper_text: str) -> str:
        return "\n\n".join([
            "=== 论文 ===",
            paper_text[:8000],
            "=== 原始题目 ===",
            state.problem_text[:2000],
            "检查要点：",
            "1. 每个子问题是否都经历 分析→建模→求解→结果→解释 的完整闭环",
            "2. 各章节之间的逻辑关系是否清晰",
            "3. 摘要是否覆盖所有子问题的结论",
            "4. 结论是否回应了题目的所有要求",
        ])

    def _parse_response(self, response: str, base: ReviewFindings) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            if any(kw in s for kw in ("缺失", "不完整", "未回答", "缺少", "断裂", "不全")):
                issues.append({"description": s.lstrip("-*# "), "severity": "warning", "category": "structure"})
            elif any(kw in s for kw in ("建议", "Suggestion")):
                suggestions.append(s.lstrip("-*# "))
        score = 80 - min(len(issues) * 10, 50)
        return ReviewFindings(
            reviewer="structure", score=max(score, 30), issues=issues or base.issues,
            suggestions=suggestions or base.suggestions, raw_report=response,
        )

    def _heuristic_structure_review(self, state: WorkflowState, paper_text: str) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []

        # check for required sections
        required = [
            ("摘要", "abstract"),
            ("关键词", "keywords"),
            ("问题重述", "restatement"),
            ("问题分析", "analysis"),
            ("模型", "model"),
            ("结果", "result"),
            ("结论", "conclusion"),
            ("参考文献", "references"),
        ]
        for cn_name, en_name in required:
            if cn_name not in paper_text and en_name not in paper_text.lower():
                issues.append({"description": f"缺少章节: {cn_name}", "severity": "warning", "category": "structure"})

        # check for evaluation/validation section
        validation_keywords = ("检验", "验证", "误差", "灵敏度", "对比", "validation", "sensitivity")
        if not any(kw in paper_text for kw in validation_keywords):
            issues.append({"description": "缺少模型检验/验证/灵敏度分析章节", "severity": "warning", "category": "structure"})

        score = 80 - min(len(issues) * 10, 50)
        return ReviewFindings(
            reviewer="structure", score=score, issues=issues, suggestions=suggestions,
            raw_report=f"启发式结构审查：{len(issues)}个问题",
        )
