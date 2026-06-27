from __future__ import annotations

import logging

from agents.base import (
    Agent,
    CodePlan,
    K_SELECTED_MODEL_IDS,
    WorkflowState,
)
from tools.prompt_loader import load_prompt

log = logging.getLogger("mma.code_plan_agent")


class CodePlanAgent(Agent):
    """Plans the code structure before generation: files, functions, model calls.

    Runs after EXPERIMENT_PLAN.  Produces ``state.code_plan``.
    User can review the plan before CodingAgent generates the actual script.
    """

    name = "code_plan_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        plan = CodePlan()

        if state.llm and state.llm.enabled:
            try:
                prompt = load_prompt("code_plan.md")
                response = state.llm.complete(prompt, self._build_llm_input(state))
                plan = self._parse_response(response, plan)
                state.code_plan = plan
                state.notes["code_plan_mode"] = "llm"
                return state
            except Exception as exc:
                state.notes["code_plan_llm_error"] = str(exc)
                log.warning("LLM code plan failed: %s", exc)

        plan = self._heuristic_plan(state)
        state.code_plan = plan
        state.notes["code_plan_mode"] = "heuristic"
        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        import json as _json
        selected_ids = []
        raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
        try:
            selected_ids = _json.loads(raw)
        except _json.JSONDecodeError:
            pass

        exp_text = ""
        if state.experiment_plan:
            ep = state.experiment_plan
            exp_text = "\n".join([
                f"指标: {ep.metrics}",
                f"数据划分: {ep.data_split}",
                f"敏感性: {ep.sensitivity_plan}",
            ])

        return "\n\n".join([
            "题目：\n" + state.problem_text.strip()[:1000],
            "已选模型ID：\n" + ", ".join(str(m) for m in selected_ids),
            "实验方案：\n" + exp_text,
            "数据文件：\n" + "\n".join(f"- {p}" for p in state.data_files),
            "\n请生成代码计划，包含：",
            "1. 需要创建的文件列表（每个文件说明用途）",
            "2. 关键函数签名（输入/输出/功能）",
            "3. 模型调用顺序与依赖关系",
        ])

    def _parse_response(self, response: str, base: CodePlan) -> CodePlan:
        files: list[dict] = []
        functions: list[dict] = []
        model_calls: list[str] = []

        current_section = ""
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            lower = s.lower()

            if "文件" in s or "file" in lower:
                current_section = "files"
                continue
            elif "函数" in s or "function" in lower:
                current_section = "functions"
                continue
            elif "模型" in s or "调用" in s or "model" in lower:
                current_section = "models"
                continue

            if current_section == "files" and s.startswith("-"):
                files.append({"path": s.lstrip("-* "), "description": ""})
            elif current_section == "functions" and s.startswith("-"):
                functions.append({"signature": s.lstrip("-* "), "description": ""})
            elif current_section == "models":
                model_calls.append(s.lstrip("-* "))

        base.files = files if files else base.files
        base.function_specs = functions if functions else base.function_specs
        base.model_calls = model_calls if model_calls else base.model_calls
        return base

    def _heuristic_plan(self, state: WorkflowState) -> CodePlan:
        import json as _json
        selected_ids = []
        raw = state.notes.get(K_SELECTED_MODEL_IDS, "[]")
        try:
            selected_ids = _json.loads(raw)
        except _json.JSONDecodeError:
            pass

        plan = CodePlan()
        plan.files = [
            {"path": "baseline_analysis.py", "description": "主执行脚本：数据加载、模型调用、结果输出"},
        ]
        plan.function_specs = [
            {"signature": "read_csv_with_fallback(path: Path) -> pd.DataFrame", "description": "多编码CSV读取"},
            {"signature": "main() -> None", "description": "入口：遍历数据文件，调用各模型，生成图表和结果表"},
        ]
        plan.model_calls = selected_ids if selected_ids else ["describe_stats", "trend_forecast", "entropy_weights", "topsis_rank"]
        return plan
