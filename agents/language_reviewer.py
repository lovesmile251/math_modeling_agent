from __future__ import annotations

import logging
import re

from agents.base import (
    A_PAPER,
    Agent,
    ReviewFindings,
    WorkflowState,
)
from tools.paper_quality import evaluate_paper_quality, format_quality_report

log = logging.getLogger("mma.language_reviewer")


class LanguageReviewerAgent(Agent):
    """Checks expression quality: abstract, structure, language, formatting.

    This is the final review gate — reuses the existing paper_quality
    evaluation and adds language-specific checks.
    """

    name = "language_reviewer"

    def run(self, state: WorkflowState) -> WorkflowState:
        findings = ReviewFindings(reviewer="language", score=80)
        paper_text = self._read_paper(state)

        if not paper_text:
            findings.issues.append({"description": "论文为空，无法审查", "severity": "high", "category": "language"})
            findings.score = 0
            state.review_findings = findings
            return state

        # existing quality evaluation
        try:
            quality = evaluate_paper_quality(
                paper_text,
                workspace_root=state.workspace.root,
                available_figures=[p.name for p in state.workspace.figures_dir.glob("*.png")],
            )
            findings.score = quality.score
            findings.suggestions = list(quality.suggestions)
            for issue in quality.issues:
                findings.issues.append({"description": issue, "severity": "warning", "category": "language"})
        except Exception as exc:
            log.warning("Quality evaluation failed: %s", exc)

        # additional language-specific checks
        lang_issues = self._check_language(paper_text)
        findings.issues.extend(lang_issues)

        state.review_findings = findings
        state.notes["language_review_score"] = str(findings.score)
        state.notes["language_review_raw"] = format_quality_report(quality) if 'quality' in dir() else ""

        return state

    def _read_paper(self, state: WorkflowState) -> str:
        pp = state.artifacts.get(A_PAPER)
        if pp and pp.exists():
            return pp.read_text(encoding="utf-8", errors="replace")
        return ""

    @staticmethod
    def _check_language(paper_text: str) -> list[dict]:
        issues: list[dict] = []

        # placeholder patterns
        placeholder_patterns = (
            r"假设为", r"待补充", r"待完善", r"待确定", r"占位",
            r"\bTODO\b", r"[xX]{3,}",
            r"记为\s*\$?[A-Z]_?\{?(?:key|a|b|c)\}?",
        )
        for pat in placeholder_patterns:
            matches = re.findall(pat, paper_text)
            if matches:
                examples = sorted(set(matches))[:5]
                issues.append({
                    "description": f"发现占位符或未完成内容: {', '.join(examples)}",
                    "severity": "warning",
                    "category": "language",
                })
                break

        # chat residue
        chatter_patterns = (
            r"^\s*好的[，,].*", r"^\s*请稍等.*", r"^\s*我将.*撰写.*",
            r"^\s*下面是.*论文.*", r"^\s*以下是.*论文.*",
        )
        for pat in chatter_patterns:
            if re.search(pat, paper_text, re.MULTILINE):
                issues.append({
                    "description": "论文包含LLM对话残留（寒暄语），请删除",
                    "severity": "warning",
                    "category": "language",
                })
                break

        return issues
