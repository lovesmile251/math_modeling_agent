from __future__ import annotations

from agents.base import Agent, WorkflowState
from tools.prompt_loader import load_prompt


class ModelingAgent(Agent):
    name = "modeling_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        if state.llm and state.llm.enabled:
            try:
                state.notes["modeling_plan"] = state.llm.complete(
                    load_prompt("modeling_plan.md"),
                    self._build_llm_input(state),
                )
                state.notes["modeling_agent_mode"] = "llm"
                return state
            except Exception as exc:
                state.notes["modeling_agent_llm_error"] = str(exc)
                state.notes["modeling_agent_mode"] = "fallback"

        has_data = bool(state.data_files)
        plan = [
            "# 建模方案",
            "",
            "## 建模目标",
            "- 将题目要求转化为可计算的指标、变量和约束。",
            "- 先建立可运行的基线模型，再根据实验结果迭代复杂模型。",
            "",
            "## 推荐流程",
            "1. 数据读取与质量检查。",
            "2. 描述性统计和可视化分析。",
            "3. 根据问题类型选择预测、优化、评价或仿真模型。",
            "4. 输出结果表、关键图表和结论解释。",
            "",
            "## 第一版模型",
        ]
        if has_data:
            plan.extend(
                [
                    "- 使用数据剖析作为基线：字段类型、缺失率、数值分布、相关性。",
                    "- 若存在明显目标列，后续可扩展为监督学习模型。",
                    "- 若存在成本、收益、容量、距离等字段，后续可扩展为优化模型。",
                ]
            )
        else:
            plan.extend(
                [
                    "- 未提供数据时，先输出符号化建模框架。",
                    "- 后续拿到数据后再生成可执行求解代码。",
                ]
            )
        plan.extend(
            [
                "",
                "## 质量检查",
                "- 代码必须可运行。",
                "- 每个结果图表必须能在论文中找到解释。",
                "- 结论必须对应题目中的具体问题。",
            ]
        )
        state.notes["modeling_plan"] = "\n".join(plan)
        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        return "\n".join(
            [
                "题目：",
                state.problem_text,
                "",
                "题目理解：",
                state.notes.get("problem_analysis", ""),
                "",
                "数据文件：",
                "\n".join(f"- {path}" for path in state.data_files) or "- 未提供数据文件",
            ]
        )
