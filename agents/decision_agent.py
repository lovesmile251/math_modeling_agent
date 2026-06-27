from __future__ import annotations

import json
import logging

from agents.base import (
    Agent,
    K_MODEL_SELECTION,
    K_SELECTED_MODEL_IDS,
    ModelCritique,
    ModelDecision,
    WorkflowState,
)
from tools.prompt_loader import load_prompt
from models.catalog import get_model_contract
from tools.model_ids import canonical_model_id, normalize_model_ids

log = logging.getLogger("mma.decision_agent")


class DecisionAgent(Agent):
    """Arbitrates between ModelingAgent proposal and ModelingCriticAgent critique.

    Produces a final ``ModelDecision`` stored in ``state.model_decision`` that
    settles on a primary model, baseline model, and the final selected model IDs.
    """

    name = "decision_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        critique = state.model_critique
        model_selection_text = state.notes.get(K_MODEL_SELECTION, "")

        if state.llm and state.llm.enabled:
            try:
                prompt = load_prompt("decision_arbitration.md")
                response = state.llm.complete(prompt, self._build_llm_input(state))
                decision = self._parse_decision(response)
                state.model_decision = decision
                state.notes["decision_agent_mode"] = "llm"
            except Exception as exc:
                state.notes["decision_agent_llm_error"] = str(exc)
                state.notes["decision_agent_mode"] = "fallback"
                log.warning("LLM decision failed, using heuristic: %s", exc)
                decision = self._heuristic_decision(state, critique)
                state.model_decision = decision
        else:
            decision = self._heuristic_decision(state, critique)
            state.model_decision = decision
            state.notes["decision_agent_mode"] = "heuristic"

        dropped = self._normalize_decision(decision)
        if dropped:
            state.notes["decision_agent_dropped_model_ids"] = json.dumps(dropped, ensure_ascii=False)

        # sync selected model IDs back to notes for downstream compatibility
        state.notes[K_SELECTED_MODEL_IDS] = json.dumps(decision.selected_model_ids, ensure_ascii=False)

        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        critique_text = ""
        if state.model_critique:
            c = state.model_critique
            critique_text = "\n".join([
                "模型批评：",
                f"风险评估：{c.risk_assessment}",
                "发现的问题：",
                *[f"- [{i.get('severity', '?')}] {i.get('description', '')}" for i in c.issues],
                "数据条件检查：",
                *[f"- {k}: {'通过' if v else '未通过'}" for k, v in c.data_condition_checks.items()],
            ])

        return "\n\n".join([
            "题目：\n" + state.problem_text.strip(),
            "建模方案：\n" + state.notes.get("modeling_plan", ""),
            "模型选择报告：\n" + state.notes.get(K_MODEL_SELECTION, ""),
            critique_text,
            "\n请基于建模方案和批评意见，做出最终决策：",
            "1. 确定主模型（primary model）",
            "2. 确定基线对照模型（baseline model）",
            "3. 列出最终选定的模型ID列表",
            "4. 给出决策理由，说明如何回应批评意见",
            "5. 输出模型对比计划",
        ])

    def _parse_decision(self, response: str) -> ModelDecision:
        decision = ModelDecision()
        lines = response.splitlines()

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            lower = line_stripped.lower()

            if "主模型" in line_stripped or "primary" in lower:
                if ":" in line_stripped:
                    decision.primary_model_id = line_stripped.split(":", 1)[1].strip().lstrip("-* ")
            elif "基线" in line_stripped or "baseline" in lower:
                if ":" in line_stripped:
                    decision.baseline_model_id = line_stripped.split(":", 1)[1].strip().lstrip("-* ")
            elif "选定模型" in line_stripped or "selected" in lower:
                if ":" in line_stripped:
                    ids_part = line_stripped.split(":", 1)[1].strip()
                    decision.selected_model_ids = [
                        s.strip().strip("'\"") for s in ids_part.replace("[", "").replace("]", "").split(",")
                        if s.strip()
                    ]

        # accumulate rationale from non-structured lines
        rationale_lines = []
        for line in lines:
            s = line.strip()
            if s and not any(kw in s.lower() for kw in ("primary", "baseline", "selected", "comparison")):
                if not s.startswith("#"):
                    rationale_lines.append(s)
        decision.rationale = "\n".join(rationale_lines[:30])

        return decision

    def _normalize_decision(self, decision: ModelDecision) -> list[str]:
        normalized = normalize_model_ids(decision.selected_model_ids)
        selected_ids = normalized.selected
        dropped = list(normalized.dropped)

        primary = canonical_model_id(decision.primary_model_id)
        if primary is None and decision.primary_model_id:
            dropped.append(decision.primary_model_id)
        baseline = canonical_model_id(decision.baseline_model_id)
        if baseline is None and decision.baseline_model_id:
            dropped.append(decision.baseline_model_id)

        if primary and primary not in selected_ids:
            selected_ids.insert(0, primary)
        if baseline and baseline not in selected_ids:
            selected_ids.append(baseline)

        decision.selected_model_ids = selected_ids
        decision.primary_model_id = primary or (selected_ids[0] if selected_ids else "")
        if baseline and baseline != decision.primary_model_id:
            decision.baseline_model_id = baseline
        elif decision.primary_model_id:
            contract = get_model_contract(decision.primary_model_id)
            decision.baseline_model_id = next(
                (model_id for model_id in contract.baseline_models if model_id in selected_ids),
                "",
            )
            if not decision.baseline_model_id:
                decision.baseline_model_id = next(
                    (model_id for model_id in selected_ids if model_id != decision.primary_model_id),
                    "",
                )
        else:
            decision.baseline_model_id = ""

        return list(dict.fromkeys(dropped))

    def _heuristic_decision(
        self, state: WorkflowState, critique: ModelCritique | None
    ) -> ModelDecision:
        """Rule-based decision when LLM is unavailable."""
        decision = ModelDecision()

        # use existing selected model IDs
        selected_raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
        try:
            selected_ids = json.loads(selected_raw)
        except json.JSONDecodeError:
            selected_ids = []

        if not isinstance(selected_ids, list):
            selected_ids = []

        # filter out high-risk models if critique has fatal issues
        high_severity_issues = [
            i for i in (critique.issues if critique else [])
            if i.get("severity") in ("high", "critical")
        ]
        if high_severity_issues and selected_ids:
            log.info("Critique found %d high-severity issues; keeping only first 2 models as safe choice.",
                     len(high_severity_issues))
            selected_ids = selected_ids[:2]

        decision.selected_model_ids = selected_ids
        decision.primary_model_id = selected_ids[0] if selected_ids else ""
        decision.baseline_model_id = ""
        if decision.primary_model_id:
            contract = get_model_contract(decision.primary_model_id)
            decision.baseline_model_id = next(
                (
                    model_id
                    for model_id in contract.baseline_models
                    if model_id in selected_ids
                ),
                "",
            )
        if not decision.baseline_model_id and len(selected_ids) > 1:
            decision.baseline_model_id = selected_ids[1]

        # rationale
        if critique and critique.issues:
            decision.rationale = (
                f"综合批评意见（{len(critique.issues)}条问题），"
                f"选定 {len(selected_ids)} 个模型，并为主模型指定可比较基线。"
            )
        else:
            decision.rationale = "基于模型选择报告的推荐，无批评意见。"

        return decision
