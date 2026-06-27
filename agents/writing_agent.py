from __future__ import annotations

import logging
import os
import re

from agents.base import (
    A_CODE,
    A_PAPER,
    A_PAPER_QUALITY,
    K_MODELING_PLAN,
    K_MODEL_SELECTION,
    K_LLM_FAILURE_KIND,
    K_PAPER_QUALITY_REPORT,
    K_PAPER_QUALITY_SCORE,
    K_PAPER_EVIDENCE_SCORE,
    K_PAPER_EXPORT_SCORE,
    K_PAPER_SOLUTION_SCORE,
    K_PAPER_STRUCTURE_SCORE,
    K_PREWRITING_GATE_REPORT,
    K_PREWRITING_GATE_STATUS,
    K_PROBLEM_ANALYSIS,
    K_PROBLEM_TYPE,
    K_RESULT_ANALYSIS,
    K_REVIEW_REPORT,
    K_WRITING_MODE,
    Agent,
    ClaimEvidence,
    ClaimEvidenceMap,
    PaperOutline,
    WorkflowState,
)
from tools.file_tool import write_text
from tools.paper_quality import clean_paper_text, evaluate_paper_quality, format_quality_report
from tools.prewriting_gate import (
    evaluate_pre_writing_gate,
    format_pre_writing_gate_report,
    write_pre_writing_gate_report,
)
from tools.prompt_loader import load_prompt
from tools.result_digest import build_result_digest
from tools.paper_templates.general import GeneralPaperTemplate
from tools.social_paper import build_social_network_paper
from tools.llm_client import classify_llm_error

log = logging.getLogger("mma.writing_agent")


PLACEHOLDER_PATTERNS = (
    r"假设为",
    r"（假设",
    r"待补充",
    r"待完善",
    r"待确定",
    r"未产出优化数值解",
    r"未产出.*数值解",
    r"未得到.*数值解",
    r"后续计算",
    r"后续可补充",
    r"后续补充",
    r"后续完善",
    r"占位",
    r"\bTODO\b",
    r"[xX]{3,}",
    r"U_\{?key\}?",
    r"U_[abc]\b",
    r"\$?U_a\$?",
    r"记为\s*\$?[A-Z]_?\{?(?:key|a|b|c)\}?",
    r"用户[A-Z]\b",
    r"某用户(?:[A-Z])?",
)


class WritingAgent(Agent):
    name = "writing_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        is_social = state.notes.get(K_PROBLEM_TYPE) == "social_network"
        gate = evaluate_pre_writing_gate(state)
        gate_path = write_pre_writing_gate_report(state, gate)
        state.artifacts["prewriting_gate"] = gate_path
        state.notes[K_PREWRITING_GATE_STATUS] = "passed" if gate.ok else "blocked"
        state.notes[K_PREWRITING_GATE_REPORT] = format_pre_writing_gate_report(gate)
        if not gate.ok:
            return self._block_for_pre_writing_gate(state, gate)

        if state.llm and state.llm.enabled:
            try:
                # ── new: section-by-section evidence-driven pipeline ──
                if self._use_fast_writing():
                    paper = self._write_single_paper(state)
                    paper = clean_paper_text(paper)
                    paper, regenerated = self._enforce_no_placeholder(state, load_prompt("paper_writing.md"), paper)
                    paper = clean_paper_text(paper)
                    paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
                    state.artifacts[A_PAPER] = paper_path
                    state.notes[K_WRITING_MODE] = "single_llm_fast"
                    state.notes["writing_agent_fast_mode"] = "true"
                    if regenerated:
                        state.notes["writing_agent_regenerated"] = "true"
                    quality = evaluate_paper_quality(paper)
                    self._record_quality(state, quality)
                    quality_path = write_text(
                        state.workspace.paper_dir / "paper_quality_report.md",
                        state.notes[K_PAPER_QUALITY_REPORT],
                    )
                    state.artifacts[A_PAPER_QUALITY] = quality_path
                    remaining = self._find_placeholders(paper)
                    if remaining:
                        state.notes["writing_agent_placeholder_warning"] = "; ".join(sorted(set(remaining)))
                    return state

                outline = self._generate_outline(state)
                state.notes["writing_agent_mode"] = "section_by_section"

                sections: dict[str, str] = {}
                total = outline.total_sections
                for i, sec in enumerate(outline.sections):
                    sec_id = sec.get("id", "")
                    log.info("Writing section %d/%d: %s", i + 1, total, sec.get("title", sec_id))
                    sections[sec_id] = self._write_section(state, sec)

                paper = self._assemble_and_unify(state, sections)
                paper = clean_paper_text(paper)
                paper, regenerated = self._enforce_no_placeholder(state, load_prompt("paper_writing.md"), paper)
                paper, refined = self._enforce_national_award_quality(state, load_prompt("paper_writing.md"), paper)
                paper = clean_paper_text(paper)
                paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
                state.artifacts[A_PAPER] = paper_path
                state.notes[K_WRITING_MODE] = "section_by_section"
                if regenerated:
                    state.notes["writing_agent_regenerated"] = "true"
                if refined:
                    state.notes["writing_agent_quality_refined"] = "true"
                quality = evaluate_paper_quality(paper)
                self._record_quality(state, quality)
                quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", state.notes[K_PAPER_QUALITY_REPORT])
                state.artifacts[A_PAPER_QUALITY] = quality_path
                paper, quality = self._apply_polish(state, paper, quality)
                remaining = self._find_placeholders(paper)
                if remaining:
                    state.notes["writing_agent_placeholder_warning"] = "; ".join(sorted(set(remaining)))
                return state
            except Exception as exc:
                state.notes["writing_agent_section_error"] = str(exc)
                failure_kind = classify_llm_error(exc)
                if failure_kind:
                    state.notes[K_LLM_FAILURE_KIND] = failure_kind
                log.warning("Section-by-section writing failed (%s), falling back to full-generation.", exc)
                state.notes[K_WRITING_MODE] = "fallback"

        if is_social:
            try:
                paper = build_social_network_paper(state.workspace, state.problem_text)
                paper = clean_paper_text(paper)
                paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
                state.artifacts[A_PAPER] = paper_path
                state.notes[K_WRITING_MODE] = "social_network_template"
                quality = evaluate_paper_quality(paper)
                self._record_quality(state, quality)
                quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", state.notes[K_PAPER_QUALITY_REPORT])
                state.artifacts[A_PAPER_QUALITY] = quality_path
                paper, quality = self._apply_polish(state, paper, quality)
                return state
            except Exception as exc:
                state.notes["writing_agent_social_error"] = str(exc)

        try:
            template = GeneralPaperTemplate(state.workspace, state.problem_text, state.notes)
            paper = template.build()
            paper = clean_paper_text(paper)
            state.notes[K_WRITING_MODE] = "general_template"
        except Exception as exc:
            state.notes["writing_agent_general_template_error"] = str(exc)
            # Ultimate fallback — should rarely trigger
            paper = "\n\n".join(
                [
                    "# 数学建模论文草稿",
                    "## 摘要\n本文基于题目要求构建了自动化数学建模工作流，完成题目理解、建模规划、代码执行、结果分析和论文草稿生成。",
                    "## 一、问题重述\n" + state.problem_text.strip(),
                    "## 二、问题分析\n" + state.notes.get(K_PROBLEM_ANALYSIS, ""),
                    "## 三、模型建立\n" + state.notes.get(K_MODELING_PLAN, ""),
                    "## 四、模型求解\n代码文件：" + str(state.artifacts.get(A_CODE, "")),
                    "## 五、结果分析\n" + state.notes.get(K_RESULT_ANALYSIS, ""),
                    "## 六、关键结果数据\n" + build_result_digest(state.workspace),
                    "## 七、审稿检查\n" + state.notes.get(K_REVIEW_REPORT, ""),
                    "## 八、模型评价与改进\n当前版本完成了基础数据剖析和论文生成闭环。后续可接入大模型推理、误差分析、灵敏度分析和模型对比。",
                    "## 附录\n完整代码见 `workspace/code/baseline_analysis.py`。",
                ]
            )
            paper = clean_paper_text(paper)
        paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
        state.artifacts[A_PAPER] = paper_path
        quality = evaluate_paper_quality(paper)
        self._record_quality(state, quality)
        quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", state.notes[K_PAPER_QUALITY_REPORT])
        state.artifacts[A_PAPER_QUALITY] = quality_path
        paper, quality = self._apply_polish(state, paper, quality)
        return state

    def _block_for_pre_writing_gate(self, state: WorkflowState, gate) -> WorkflowState:
        paper = "\n\n".join(
            [
                "# 写作前证据门禁未通过",
                "",
                "当前运行尚不具备生成最终论文的证据条件，因此系统停止论文正文写作，避免编造模型结果。",
                "",
                state.notes[K_PREWRITING_GATE_REPORT],
            ]
        )
        paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
        state.artifacts[A_PAPER] = paper_path
        quality = evaluate_paper_quality(paper)
        self._record_quality(state, quality)
        quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", state.notes[K_PAPER_QUALITY_REPORT])
        state.artifacts[A_PAPER_QUALITY] = quality_path
        state.notes[K_WRITING_MODE] = "prewriting_gate_blocked"
        return state

    def _record_quality(self, state: WorkflowState, quality) -> None:
        state.notes[K_PAPER_QUALITY_SCORE] = str(quality.score)
        state.notes[K_PAPER_QUALITY_REPORT] = format_quality_report(quality)
        state.notes[K_PAPER_SOLUTION_SCORE] = str(quality.metrics.get("solution_score", quality.score))
        state.notes[K_PAPER_EVIDENCE_SCORE] = str(quality.metrics.get("evidence_score", quality.score))
        state.notes[K_PAPER_STRUCTURE_SCORE] = str(quality.metrics.get("structure_score", quality.score))
        state.notes[K_PAPER_EXPORT_SCORE] = str(quality.metrics.get("export_score", quality.score))

    # ── section-by-section evidence-driven writing pipeline ────────────
    def _use_fast_writing(self) -> bool:
        mode = os.environ.get("MMA_LLM_WRITING_MODE", "").strip().lower()
        fast = os.environ.get("MMA_LLM_FAST_MODE", "").strip().lower()
        return mode in {"single", "fast"} or fast in {"1", "true", "yes", "on"}

    def _write_single_paper(self, state: WorkflowState) -> str:
        instructions = load_prompt("paper_writing.md")
        input_text = "\n\n".join(
            [
                self._build_llm_input(state),
                "写作模式：快速单次生成。",
                "请一次性生成完整 Markdown 论文，包含摘要、关键词、问题重述、问题分析、模型假设、模型建立与求解、结果分析、模型检验/对比、结论和参考文献。",
                "只使用输入中出现的真实结果、表格、图和模型名称；不要输出过程说明。",
            ]
        )
        return state.llm.complete(instructions, input_text)

    def _generate_outline(self, state: WorkflowState) -> PaperOutline:
        """Generate a structured paper outline with evidence pre-assignment."""
        outline = PaperOutline()
        evidence_claims = state.claim_evidence_map.claims if state.claim_evidence_map else []

        # default sections for math modeling paper
        default_sections = [
            {"id": "abstract", "title": "摘要", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "keywords", "title": "关键词", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "restatement", "title": "一、问题重述", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "analysis", "title": "二、问题分析", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "assumptions", "title": "三、模型假设与符号说明", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "model", "title": "四、模型建立与求解", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "results", "title": "五、结果分析", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "validation", "title": "六、模型检验与对比", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "conclusion", "title": "七、结论", "available_claims": [], "available_figures": [], "available_tables": []},
            {"id": "references", "title": "参考文献", "available_claims": [], "available_figures": [], "available_tables": []},
        ]

        # pre-assign evidence to sections
        tables = sorted(state.workspace.tables_dir.glob("*.csv"))
        figures = sorted(state.workspace.figures_dir.glob("*.png"))

        for section in default_sections:
            if section["id"] in ("abstract", "conclusion"):
                # abstract/conclusion can reference any claim
                section["available_claims"] = [c.claim_id for c in evidence_claims]
            elif section["id"] == "results":
                section["available_claims"] = [c.claim_id for c in evidence_claims]
                section["available_tables"] = [t.stem for t in tables]
                section["available_figures"] = [f.stem for f in figures]
            elif section["id"] == "validation":
                section["available_claims"] = [c.claim_id for c in evidence_claims
                                              if any(kw in c.claim.lower() for kw in ("对比", "误差", "检验", "comparison", "error"))]
            elif section["id"] == "model":
                section["available_tables"] = [t.stem for t in tables]

        outline.sections = default_sections
        outline.total_sections = len(default_sections)
        state.paper_outline = outline
        return outline

    def _write_section(self, state: WorkflowState, section: dict) -> str:
        """Write one paper section using assigned evidence only."""
        section_id = section.get("id", "")
        title = section.get("title", "")

        if state.llm and state.llm.enabled:
            try:
                instructions = load_prompt("paper_writing.md")
                input_text = self._build_section_input(state, section)
                return state.llm.complete(instructions, input_text)
            except Exception:
                pass

        # heuristic fallback for specific sections
        if section_id == "abstract":
            return self._heuristic_abstract(state)
        elif section_id == "conclusion":
            return self._heuristic_conclusion(state)
        elif section_id == "results":
            return f"## {title}\n\n" + state.notes.get(K_RESULT_ANALYSIS, "")
        else:
            return f"## {title}\n\n（本节待生成）"

    def _build_section_input(self, state: WorkflowState, section: dict) -> str:
        section_id = section.get("id", "")
        title = section.get("title", "")
        claim_ids = section.get("available_claims", [])

        evidence_text = ""
        if state.claim_evidence_map:
            relevant = [c for c in state.claim_evidence_map.claims if c.claim_id in claim_ids]
            evidence_text = "\n".join(
                f"[{c.claim_id}] {c.claim} (来源: {c.source_file}, 行: {c.source_rows})"
                for c in relevant[:15]
            )

        return "\n\n".join([
            f"=== 写作任务：{title} ===",
            "题目：\n" + state.problem_text.strip()[:2000],
            "问题分析：\n" + state.notes.get(K_PROBLEM_ANALYSIS, "")[:1000],
            "建模方案：\n" + state.notes.get(K_MODELING_PLAN, "")[:1000],
            "结果分析：\n" + state.notes.get(K_RESULT_ANALYSIS, "")[:2000],
            f"=== 本节可用证据（claim_id 列表）===\n{', '.join(claim_ids) if claim_ids else '无'}",
            f"=== 证据详情 ===\n{evidence_text or '无'}",
            f"=== 可用图表 ===\n{', '.join(section.get('available_figures', []))}",
            f"=== 可用表格 ===\n{', '.join(section.get('available_tables', []))}",
            "约束：",
            "- 只能使用上面列出的证据和图表，不得编造",
            "- 每个数值必须对应一个 claim_id",
            "- 无证据的结论标注'数据不足，后续可补充'",
        ])

    def _assemble_and_unify(self, state: WorkflowState, sections: dict[str, str]) -> str:
        """Assemble sections into a final paper with unified numbering and cross-references."""
        outline = state.paper_outline
        if not outline:
            return "\n\n".join(sections.values())

        parts: list[str] = []
        for sec in outline.sections:
            sec_id = sec.get("id", "")
            content = sections.get(sec_id, "")
            if content:
                # clean each section
                content = clean_paper_text(content)
                parts.append(content)

        paper = "\n\n".join(parts)

        # unified numbering pass
        paper = self._unify_numbering(paper)

        # evidence enforcement: check abstract and conclusion
        paper = self._enforce_evidence_coverage(state, paper)

        return paper

    def _unify_numbering(self, paper: str) -> str:
        """Re-number equations and figures sequentially."""
        import re

        eq_counter = [0]
        def _renumber_eq(m):
            eq_counter[0] += 1
            return f"({eq_counter[0]})"

        # unify equation numbers to (1), (2), ...
        paper = re.sub(r"\\tag\{[^}]*\}|\\qquad\(\d+\)|\(\d+\)", _renumber_eq, paper)
        return paper

    def _enforce_evidence_coverage(self, state: WorkflowState, paper: str) -> str:
        """Ensure every numerical claim in abstract/conclusion has evidence."""
        if not state.claim_evidence_map:
            return paper

        cem = state.claim_evidence_map
        unmapped_in_abstract = []

        # extract abstract and conclusion sections
        import re
        abstract_match = re.search(r"(?:摘要|abstract).*?(?=关键词|keywords|一、|1\.|##)", paper, re.DOTALL | re.IGNORECASE)
        conclusion_match = re.search(r"(?:结论|conclusion).*", paper, re.DOTALL | re.IGNORECASE)

        for section_text in [abstract_match.group(0) if abstract_match else "", conclusion_match.group(0) if conclusion_match else ""]:
            # find all numbers
            numbers = re.findall(r"\d+\.?\d*", section_text)
            for num in numbers:
                # check if this number appears in any claim
                found = any(num in c.claim or num in c.calculation for c in cem.claims)
                if not found and len(num) > 2:  # ignore small integers like 1, 2
                    unmapped_in_abstract.append(num)

        if unmapped_in_abstract:
            state.notes["evidence_enforcement_warning"] = (
                f"摘要/结论中发现 {len(unmapped_in_abstract)} 个未映射的数值: {', '.join(unmapped_in_abstract[:10])}"
            )
            log.warning(state.notes["evidence_enforcement_warning"])

        return paper

    def _heuristic_abstract(self, state: WorkflowState) -> str:
        claims = state.claim_evidence_map.claims if state.claim_evidence_map else []
        claim_lines = "\n".join(f"- {c.claim}" for c in claims[:5]) if claims else "- 暂无具体数值结果"
        return "\n".join([
            "## 摘要",
            f"本文针对'{state.problem_text.strip()[:80]}...'问题，建立了数学模型进行分析。",
            "",
            "**关键结果：**",
            claim_lines,
            "",
            "**方法：** " + (state.notes.get(K_MODELING_PLAN, "")[:200] or "基于数据分析和建模方法"),
        ])

    def _heuristic_conclusion(self, state: WorkflowState) -> str:
        return "\n".join([
            "## 七、结论",
            "本文完成了从问题分析到模型求解的完整建模流程。",
            "详细结果见上文各章节。",
        ])
        result_digest = build_result_digest(state.workspace)

        # Fetch real references based on selected models
        import json as _json
        from tools.reference_fetcher import fetch_references, format_references_section
        model_ids_raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
        try:
            model_ids = _json.loads(model_ids_raw) if model_ids_raw else []
        except _json.JSONDecodeError:
            model_ids = []
        references = fetch_references(
            selected_models=list(model_ids) if isinstance(model_ids, list) else [],
            problem_text=state.problem_text,
            min_count=8,
            max_count=12,
        )
        refs_text = format_references_section(references)

        # Build innovation suggestions from model selection report
        innovation_text = self._build_innovation_input(state)

        # Build model comparison requirement
        comparison_text = self._build_comparison_input(state)

        return "\n\n".join(
            [
                "题目：\n" + state.problem_text.strip(),
                "问题分析：\n" + state.notes.get(K_PROBLEM_ANALYSIS, ""),
                "建模方案：\n" + state.notes.get(K_MODELING_PLAN, ""),
                "模型选择报告：\n" + state.notes.get(K_MODEL_SELECTION, ""),
                "结果分析：\n" + state.notes.get(K_RESULT_ANALYSIS, ""),
                "===== 真实结果数据（必须据此写出具体数值结论，下面没有的数值不得编造）=====\n" + result_digest,
                "审稿检查：\n" + state.notes.get(K_REVIEW_REPORT, ""),
                "代码文件：\n" + str(state.artifacts.get(A_CODE, "")),
                "可引用的已生成结果文件：\n" + "\n".join(f"- {name}: {path}" for name, path in state.artifacts.items()),
                "===== 模型创新点（必须在论文中体现，并结合题目说明创新价值）=====\n" + innovation_text,
                "===== 模型对比要求（必须生成模型对比表格）=====\n" + comparison_text,
                "===== 真实参考文献（必须在正文中引用，不得编造）=====\n" + refs_text,
                "写作约束：\n"
                "1. 摘要、结果分析、结论中的每个数值、排名、参数、分类/聚类结果都必须能在上面的\"真实结果数据\"中找到对应来源；找不到就不要写。\n"
                "2. 严禁使用 U_a、U_b、$U_{key}$、\"假设为\"、\"记为 X\"、\"待补充\"等占位符或虚构对象，要用真实结果中的实际取值。\n"
                "3. 不得声称使用了未在结果数据中出现的模型（如未运行就不要写 Lasso、随机森林、神经网络、遗传算法、ARIMA、Prophet、XGBoost、LightGBM、Louvain 等）。\n"
                "4. 引用图表时只能引用上面\"可在正文引用的图表文件\"中列出的文件名。\n"
                "5. 必须在正文中恰当位置引用上述真实参考文献（用 [1][2] 等上标标注），参考文献列表必须与引用一一对应，不得编造文献。\n"
                "6. 每个模型章节必须包含：为什么不用更简单模型、为什么不用更复杂模型、当前模型适用的数据条件、模型假设、参数设置、验证指标、局限性、改进模型带来的提升。\n"
                "7. 必须包含\"模型创新点\"章节，逐一说明本文采用的方法创新及其价值；必须包含\"模型对比\"章节，以表格形式列出各候选模型的对比指标。",            ]
        )

    def _build_llm_input(self, state: WorkflowState) -> str:
        result_digest = build_result_digest(state.workspace)
        artifacts = "\n".join(f"- {name}: {path}" for name, path in state.artifacts.items()) or "- none"
        tables = "\n".join(f"- {path.name}" for path in sorted(state.workspace.tables_dir.glob("*.csv"))[:30]) or "- none"
        figures = "\n".join(f"- {path.name}" for path in sorted(state.workspace.figures_dir.glob("*.png"))[:30]) or "- none"
        claims = ""
        if state.claim_evidence_map:
            claims = "\n".join(f"- [{c.claim_id}] {c.claim}" for c in state.claim_evidence_map.claims[:20])

        return "\n\n".join(
            [
                "题目：\n" + state.problem_text.strip()[:3000],
                "问题分析：\n" + state.notes.get(K_PROBLEM_ANALYSIS, "")[:1500],
                "建模方案：\n" + state.notes.get(K_MODELING_PLAN, "")[:2000],
                "模型选择报告：\n" + state.notes.get(K_MODEL_SELECTION, "")[:2000],
                "结果分析：\n" + state.notes.get(K_RESULT_ANALYSIS, "")[:2500],
                "真实结果数据：\n" + result_digest[:5000],
                "证据声明：\n" + (claims or "none"),
                "可引用表格：\n" + tables,
                "可引用图：\n" + figures,
                "已有产物：\n" + artifacts,
                "审稿检查：\n" + state.notes.get(K_REVIEW_REPORT, "")[:1500],
            ]
        )

    def _enforce_no_placeholder(self, state: WorkflowState, instructions: str, paper: str) -> tuple[str, bool]:
        offenders = self._find_placeholders(paper)
        if not offenders:
            return paper, False

        correction = (
            "上一稿存在以下占位符或疑似虚构内容，必须修正：\n"
            + "\n".join(f"- {snippet}" for snippet in sorted(set(offenders)))
            + "\n\n请重写完整论文：删除所有占位符，所有数值结论改用下面“真实结果数据”中的实际取值；"
            "若真实结果数据不足以支撑某个结论，就删除该结论或改写为已完成数据范围内的保守结论；"
            "不得写“待补充”“后续计算”“未产出优化数值解”等不可提交表述。\n\n"
            + self._build_llm_input(state)
        )
        try:
            revised = state.llm.complete(instructions, correction)
        except Exception as exc:
            state.notes["writing_agent_regenerate_error"] = str(exc)
            failure_kind = classify_llm_error(exc)
            if failure_kind:
                state.notes[K_LLM_FAILURE_KIND] = failure_kind
            return paper, False
        return revised, True

    def _enforce_national_award_quality(
        self, state: WorkflowState, instructions: str, paper: str
    ) -> tuple[str, bool]:
        quality = evaluate_paper_quality(paper)
        if quality.score >= 82:
            return paper, False

        try:
            deep_study = load_prompt("national_award_paper_deep_study.md")
        except Exception:
            deep_study = ""

        refinement_input = "\n\n".join(
            [
                "上一稿未达到国奖论文质量门禁，必须在不编造数据的前提下重写增强。",
                format_quality_report(quality),
                "国奖论文逐篇研究结论：\n" + deep_study[:6000],
                "原论文草稿：\n" + paper,
                "真实结果数据与写作约束：\n" + self._build_llm_input(state),
                "重写要求：\n"
                "1. 删除所有非论文寒暄和生成说明。\n"
                "2. 摘要逐题给模型、真实数值和结论意义。\n"
                "3. 每个子问题必须按分析、模型、求解、结果、解释形成闭环。\n"
                "4. 补充模型检验、误差分析、基准对比或灵敏度分析；如果结果数据不足以支撑某个结论，删除该结论或限定为已有数据支持的结论。\n"
                "5. 正文引用每张关键图表并解释图表含义。\n"
                "6. 保持 Markdown 格式，公式用 \\( \\) 或 \\[ \\]，不要输出说明性前言。\n"
                "7. 不得保留“待补充”“后续计算”“未产出优化数值解”等不可提交表述。",
            ]
        )
        try:
            revised = state.llm.complete(instructions, refinement_input)
        except Exception as exc:
            state.notes["writing_agent_quality_refine_error"] = str(exc)
            failure_kind = classify_llm_error(exc)
            if failure_kind:
                state.notes[K_LLM_FAILURE_KIND] = failure_kind
            return paper, False
        return clean_paper_text(revised), True

    def _find_placeholders(self, paper: str) -> list[str]:
        found: list[str] = []
        for pattern in PLACEHOLDER_PATTERNS:
            for match in re.finditer(pattern, paper):
                snippet = match.group(0).strip()
                if snippet:
                    found.append(snippet)
        return found

    def _build_innovation_input(self, state: WorkflowState) -> str:
        """Extract innovation suggestions from model selection report for the LLM prompt."""
        import json as _json
        try:
            report_path = state.artifacts.get("model_selection_report")
            if report_path:
                with open(report_path, "r", encoding="utf-8") as fh:
                    report = _json.load(fh)
                innovations = report.get("innovation_recommendations", [])
                if innovations:
                    lines = ["以下是从模型选择中识别到的创新建议，请融入论文："]
                    for inn in innovations:
                        lines.append(f"- {inn.get('label', inn.get('model_id', ''))} (tier={inn.get('tier','')}, 创新评分={inn.get('innovation_score','')}/10): {inn.get('reason','')}")
                        exts = inn.get("innovation_extensions", [])
                        if exts:
                            lines.append(f"  可用的创新扩展: {', '.join(exts)}")
                    return "\n".join(lines)
            return "（无创新建议记录，请根据模型选择报告中的tier信息自行提炼）"
        except Exception:
            return "（无法读取创新建议）"

    def _build_comparison_input(self, state: WorkflowState) -> str:
        """Build model comparison requirements for the LLM prompt."""
        import json as _json
        try:
            report_path = state.artifacts.get("model_selection_report")
            if report_path:
                with open(report_path, "r", encoding="utf-8") as fh:
                    report = _json.load(fh)
                comparison_plan = report.get("model_comparison_plan", [])
                if comparison_plan:
                    lines = ["以下模型需要生成对比表格（按任务类型分组）："]
                    for plan in comparison_plan:
                        mid = plan.get("model_id", "")
                        label = plan.get("label", "")
                        task = plan.get("task_type", "")
                        metrics = plan.get("metrics", [])
                        comps = plan.get("comparison_candidates", [])
                        lines.append(f"- {label} ({mid}) — 任务类型: {task}")
                        lines.append(f"  对比指标: {', '.join(metrics[:5])}")
                        if comps:
                            lines.append(f"  对比候选: {', '.join(comps)}")
                    return "\n".join(lines)
            return "（无对比计划，请根据已选模型自行设计对比表格，至少对比基线模型和改进模型）"
        except Exception:
            return "（无法读取对比计划，请自行设计模型对比表格）"

    def _polish_paper(self, state: WorkflowState, paper: str) -> str:
        """Content-aware polish: verify data consistency + fix language."""
        # Step 1: grammar/style polish (existing)
        prompt1 = (
            "Polish this academic paper: fix grammar, improve flow, ensure consistent terminology. "
            "Keep all numbers and data unchanged. Return the polished paper."
        )
        try:
            paper = state.llm.complete(prompt1, paper)
        except Exception as exc:
            failure_kind = classify_llm_error(exc)
            if failure_kind:
                state.notes[K_LLM_FAILURE_KIND] = failure_kind
            return paper

        # Step 2: content consistency check
        result_digest = build_result_digest(state.workspace)
        prompt2 = (
            "You are a rigorous fact-checker. Compare the abstract of this paper against the results section and the raw data below. "
            "List any discrepancies where:\n"
            "1. A number in the abstract does NOT appear in the results\n"
            "2. A claimed model name was not actually run\n"
            "3. A ranking or comparison contradicts the data\n"
            "4. A figure is referenced that doesn't exist\n\n"
            f"=== RAW DATA ===\n{result_digest[:4000]}\n\n"
            f"=== PAPER ===\n{paper[:8000]}\n\n"
            "If you find issues, fix them in the paper. If no issues, return the paper unchanged. "
            "Return ONLY the corrected paper, no explanations."
        )
        try:
            return state.llm.complete(prompt2, paper)
        except Exception as exc:
            failure_kind = classify_llm_error(exc)
            if failure_kind:
                state.notes[K_LLM_FAILURE_KIND] = failure_kind
            return paper

    def _apply_polish(self, state: WorkflowState, paper: str, quality) -> tuple[str, object]:
        """Optionally polish the paper via LLM and update state artifacts.

        Only runs when state.llm is enabled and quality.score < 90.
        Returns (paper, quality) — possibly updated.
        """
        if not (state.llm and state.llm.enabled and quality.score < 90):
            return paper, quality

        try:
            polished = self._polish_paper(state, paper)
            paper = clean_paper_text(polished)
            paper_path = write_text(state.workspace.paper_dir / "paper_draft.md", paper)
            state.artifacts[A_PAPER] = paper_path
            quality = evaluate_paper_quality(paper)
            self._record_quality(state, quality)
            quality_path = write_text(state.workspace.paper_dir / "paper_quality_report.md", state.notes[K_PAPER_QUALITY_REPORT])
            state.artifacts[A_PAPER_QUALITY] = quality_path
            state.notes["writing_agent_polished"] = "true"
        except Exception as exc:
            state.notes["writing_agent_polish_error"] = str(exc)
            failure_kind = classify_llm_error(exc)
            if failure_kind:
                state.notes[K_LLM_FAILURE_KIND] = failure_kind

        return paper, quality
