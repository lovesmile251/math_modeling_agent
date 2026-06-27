from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from agents.base import (
    A_PAPER,
    A_TRACEABILITY_REPORT,
    Agent,
    ClaimEvidenceMap,
    ReviewFindings,
    WorkflowState,
)
from tools.traceability import evaluate_numeric_traceability, write_traceability_report

log = logging.getLogger("mma.fact_reviewer")


class FactReviewerAgent(Agent):
    """Checks factual consistency: every numerical claim must be traceable
    to a source file, row, and calculation in the ClaimEvidenceMap.

    Depends on EvidenceAgent having populated ``state.claim_evidence_map``.
    """

    name = "fact_reviewer"

    def run(self, state: WorkflowState) -> WorkflowState:
        findings = ReviewFindings(reviewer="fact", score=80)
        paper_text = self._read_paper(state)

        if state.llm and state.llm.enabled and paper_text:
            try:
                from tools.prompt_loader import load_prompt
                prompt = load_prompt("fact_review.md")
                response = state.llm.complete(prompt, self._build_llm_input(state, paper_text))
                findings = self._parse_response(response, findings)
                state.notes["fact_review_score"] = str(findings.score)
            except Exception as exc:
                state.notes["fact_review_error"] = str(exc)
                log.warning("LLM fact review failed: %s", exc)
                findings = self._heuristic_fact_review(state, paper_text)
        else:
            findings = self._heuristic_fact_review(state, paper_text)

        traceability = evaluate_numeric_traceability(
            paper_text,
            state.workspace,
            state.claim_evidence_map,
        )
        traceability_path = write_traceability_report(state.workspace, traceability)
        state.artifacts[A_TRACEABILITY_REPORT] = traceability_path
        state.notes["traceability_gate"] = "passed" if traceability.passed else "failed"
        state.notes["traceability_coverage_pct"] = str(traceability.coverage_pct)
        if not traceability.passed:
            findings.issues.append(
                {
                    "description": (
                        f"数值结论追溯覆盖率仅 {traceability.coverage_pct:.1f}%，"
                        f"低于 {traceability.threshold_pct:.0f}% 门槛"
                    ),
                    "severity": "high",
                    "category": "traceability",
                }
            )
            findings.score = max(20, findings.score - 20)
        state.review_findings = findings
        state.notes["fact_review_raw"] = findings.raw_report
        return state

    def _read_paper(self, state: WorkflowState) -> str:
        pp = state.artifacts.get(A_PAPER)
        if pp and pp.exists():
            return pp.read_text(encoding="utf-8", errors="replace")
        return ""

    def _build_llm_input(self, state: WorkflowState, paper_text: str) -> str:
        evidence_text = ""
        if state.claim_evidence_map:
            cem = state.claim_evidence_map
            evidence_text = "\n".join(
                f"[{c.claim_id}] {c.claim} ← {Path(c.source_file).name}:{c.source_rows}"
                for c in cem.claims[:20]
            )
        return "\n\n".join([
            "=== 论文 ===",
            paper_text[:8000],
            "=== 证据映射 ===",
            evidence_text or "（无证据映射）",
            "检查要点：",
            "1. 论文中的每个数值是否都有对应的证据来源",
            "2. 是否存在没有结果文件支撑的模型声明",
            "3. 排名、百分比、对比数据是否与实际结果一致",
            "4. 图表引用是否正确",
        ])

    def _parse_response(self, response: str, base: ReviewFindings) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            if any(kw in s for kw in ("无来源", "未找到", "不匹配", "虚构", "不一致", "找不到")):
                issues.append({"description": s.lstrip("-*# "), "severity": "high", "category": "fact"})
            elif any(kw in s for kw in ("建议", "Suggestion", "可补充")):
                suggestions.append(s.lstrip("-*# "))
        score = 80 - min(len(issues) * 15, 60)
        return ReviewFindings(
            reviewer="fact", score=max(score, 20), issues=issues or base.issues,
            suggestions=suggestions or base.suggestions, raw_report=response,
        )

    def _heuristic_fact_review(self, state: WorkflowState, paper_text: str) -> ReviewFindings:
        issues: list[dict] = []
        suggestions: list[str] = []

        # check evidence map coverage
        cem = state.claim_evidence_map
        if cem and cem.coverage_pct < 50:
            issues.append({"description": f"证据覆盖率仅 {cem.coverage_pct:.0f}%，论文中大量数值可能缺少溯源", "severity": "high", "category": "fact"})
        elif cem and cem.claims:
            suggestions.append(f"证据映射包含 {len(cem.claims)} 条声明，覆盖率 {cem.coverage_pct:.0f}%")
        else:
            issues.append({"description": "缺少证据映射，无法验证论文数值的真实性", "severity": "high", "category": "fact"})

        # check for unsupported model claims
        run_summary = state.workspace.logs_dir / "run_summary.json"
        models_in_results: set[str] = set()
        if run_summary.exists():
            try:
                data = json.loads(run_summary.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            models_in_results.update(item.get("selected_models", []))
            except (json.JSONDecodeError, OSError):
                pass

        common_models = {"random_forest", "neural_network", "xgboost", "lightgbm", "lstm", "arima", "prophet", "genetic_algorithm"}
        mentioned_in_paper = {m for m in common_models if m in paper_text.lower() or m.replace("_", " ") in paper_text.lower()}
        unsupported = mentioned_in_paper - models_in_results
        if unsupported:
            issues.append({"description": f"论文提及但未实际运行的模型: {', '.join(sorted(unsupported))}", "severity": "high", "category": "fact"})

        score = 80 - min(len([i for i in issues if i.get("severity") == "high"]) * 20, 60)
        return ReviewFindings(
            reviewer="fact", score=score, issues=issues, suggestions=suggestions,
            raw_report=f"启发式事实审查：{len(issues)}个问题",
        )
