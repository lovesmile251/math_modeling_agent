from __future__ import annotations

import logging

from agents.base import (
    Agent,
    K_MODELING_PLAN,
    K_MODEL_SELECTION,
    K_PROBLEM_ANALYSIS,
    K_SELECTED_MODEL_IDS,
    ModelCritique,
    WorkflowState,
)
from tools.prompt_loader import load_prompt

log = logging.getLogger("mma.modeling_critic_agent")


class ModelingCriticAgent(Agent):
    """Reviews model proposals for data-condition fit, overfitting risk, and assumption validity.

    This agent acts as the "critic" in a propose→critique→decide pattern.
    It reads the modeling plan and model selection report, then produces a
    structured ModelCritique stored in ``state.model_critique``.
    """

    name = "modeling_critic_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        critique = ModelCritique()

        modeling_plan = state.notes.get(K_MODELING_PLAN, "")
        model_selection = state.notes.get(K_MODEL_SELECTION, "")
        problem_analysis = state.notes.get(K_PROBLEM_ANALYSIS, "")

        if state.llm and state.llm.enabled:
            try:
                prompt = load_prompt("modeling_critic.md")
                response = state.llm.complete(prompt, self._build_llm_input(state))
                critique = self._parse_llm_response(response, critique)
                state.model_critique = critique
                state.notes["modeling_critic_mode"] = "llm"
                return state
            except Exception as exc:
                state.notes["modeling_critic_llm_error"] = str(exc)
                state.notes["modeling_critic_mode"] = "fallback"
                log.warning("LLM critique failed, using heuristic fallback: %s", exc)

        # heuristic fallback
        critique = self._heuristic_critique(state, modeling_plan, model_selection, problem_analysis)
        state.model_critique = critique
        state.notes["modeling_critic_mode"] = "heuristic"
        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        import json as _json
        selected_ids_raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
        try:
            selected_ids = _json.loads(selected_ids_raw)
        except _json.JSONDecodeError:
            selected_ids = []

        return "\n\n".join([
            "题目：\n" + state.problem_text.strip(),
            "问题分析：\n" + state.notes.get(K_PROBLEM_ANALYSIS, ""),
            "建模方案：\n" + state.notes.get(K_MODELING_PLAN, ""),
            "模型选择报告：\n" + state.notes.get(K_MODEL_SELECTION, ""),
            "已选模型ID：\n" + ", ".join(str(m) for m in selected_ids),
            "数据文件：\n" + "\n".join(f"- {p}" for p in state.data_files),
            "\n请从以下维度审视模型方案：",
            "1. 每个候选模型的数据条件是否满足（正态性、线性、独立性、样本量等）",
            "2. 是否存在过拟合风险（模型复杂度过高、参数过多、样本不足）",
            "3. 模型假设是否与题目实际场景一致",
            "4. 是否遗漏了更简单或更合适的替代模型",
            "5. 对照模型是否合理",
        ])

    def _parse_llm_response(self, response: str, base: ModelCritique) -> ModelCritique:
        """Parse LLM response into structured critique."""
        issues: list[dict] = []
        conditions: dict[str, bool] = {}
        risk_text = ""

        current_section = ""
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or "风险" in line:
                current_section = "risk"
                continue
            if "数据条件" in line or "条件检查" in line:
                current_section = "conditions"
                continue
            if "问题" in line or "issue" in line.lower():
                current_section = "issues"
                continue

            if current_section == "risk" and len(risk_text) < 2000:
                risk_text += line + "\n"
            elif current_section == "conditions":
                # try to parse "model_name: satisfied/unsatisfied"
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip().lstrip("-* ")
                    val = "true" in parts[1].lower() or "ok" in parts[1].lower() or "满足" in parts[1]
                    conditions[key] = val
            elif current_section == "issues":
                issues.append({"description": line.lstrip("-* "), "severity": "warning"})

        base.issues = issues if issues else base.issues
        base.data_condition_checks = conditions if conditions else base.data_condition_checks
        base.risk_assessment = risk_text.strip() if risk_text else base.risk_assessment
        return base

    def _heuristic_critique(
        self,
        state: WorkflowState,
        modeling_plan: str,
        model_selection: str,
        problem_analysis: str,
    ) -> ModelCritique:
        """Rule-based critique when LLM is unavailable."""
        critique = ModelCritique()
        issues: list[dict] = []
        conditions: dict[str, bool] = {}

        # check data availability
        if not state.data_files:
            issues.append({
                "description": "未提供数据文件，无法验证模型的数据条件是否满足",
                "severity": "high",
            })
        else:
            # check sample size vs model complexity
            import json as _json
            selected_ids_raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
            try:
                selected_ids = _json.loads(selected_ids_raw)
            except _json.JSONDecodeError:
                selected_ids = []

            model_count = len(selected_ids)
            if model_count > 5:
                issues.append({
                    "description": f"选择了 {model_count} 个模型，可能存在过拟合风险，建议精简至 2-4 个核心模型",
                    "severity": "warning",
                })
            if model_count == 0:
                issues.append({
                    "description": "未选择任何可执行模型，论文将缺少实质性建模内容",
                    "severity": "high",
                })

        # check modeling plan content
        if len(modeling_plan.strip()) < 100:
            issues.append({
                "description": "建模方案内容过短，可能缺少详细的模型假设和求解思路",
                "severity": "warning",
            })
        if len(model_selection.strip()) < 50:
            issues.append({
                "description": "模型选择报告内容过短，缺少模型对比和选择理由",
                "severity": "warning",
            })

        # keyword-based assumption checks
        plan_lower = modeling_plan.lower() + model_selection.lower() + problem_analysis.lower()
        if "假设" not in plan_lower:
            issues.append({
                "description": "建模方案中未明确列出模型假设，建议补充独立性、分布假设等",
                "severity": "warning",
            })
        if "检验" not in plan_lower and "验证" not in plan_lower:
            issues.append({
                "description": "未提及模型检验或验证方案，建议补充残差分析、交叉验证等",
                "severity": "warning",
            })

        # data condition checks
        for keyword, check_name in [
            ("正态", "normality_check"),
            ("线性", "linearity_check"),
            ("独立", "independence_check"),
        ]:
            conditions[check_name] = keyword in plan_lower

        if not issues:
            issues.append({
                "description": "基础检查通过，建议在获得实验结果后进一步验证模型假设",
                "severity": "info",
            })

        critique.issues = issues
        critique.data_condition_checks = conditions
        critique.risk_assessment = "启发式评估：基于关键词和规则检查，未使用LLM深度分析。"

        return critique
