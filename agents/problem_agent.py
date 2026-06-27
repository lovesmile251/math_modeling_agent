from __future__ import annotations

import json
import re

from agents.base import A_PROBLEM_SPEC, Agent, ProblemSpec, WorkflowState
from agents.model_selection_crew import DataProfileAgent, TaskDecompositionAgent
from tools.data_profiler import summarize_data_files
from tools.file_tool import write_text
from tools.prompt_loader import load_prompt


class ProblemAgent(Agent):
    name = "problem_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        tasks = TaskDecompositionAgent(llm=state.llm).run(
            state.problem_text,
            self._data_columns(state),
        )
        profile = DataProfileAgent().run(state.data_files)

        if state.llm and state.llm.enabled:
            try:
                state.notes["problem_analysis"] = state.llm.complete(
                    load_prompt("problem_analysis.md"),
                    self._build_llm_input(state),
                )
                state.notes["problem_agent_mode"] = "llm"
            except Exception as exc:
                state.notes["problem_agent_llm_error"] = str(exc)
                state.notes["problem_agent_mode"] = "fallback"

        spec = self._build_problem_spec(state, tasks, profile)
        state.problem_spec = spec
        spec_path = write_text(
            state.workspace.logs_dir / "problem_spec.json",
            json.dumps(spec.__dict__, ensure_ascii=False, indent=2),
        )
        state.artifacts[A_PROBLEM_SPEC] = spec_path

        if "problem_analysis" not in state.notes:
            state.notes["problem_analysis"] = self._format_analysis(spec, state)
            state.notes["problem_agent_mode"] = "structured_fallback"
        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        data_lines = "\n".join(f"- {path}" for path in state.data_files) or "- 未提供数据文件"
        return "\n".join(
            [
                "题目：",
                state.problem_text,
                "",
                "数据文件：",
                data_lines,
            ]
        )

    def _build_problem_spec(self, state, tasks, profile) -> ProblemSpec:
        text = state.problem_text.strip()
        subproblems: list[dict] = []
        for task in tasks:
            subproblems.append(
                {
                    "id": task.task_id,
                    "task_type": task.task_type,
                    "objective": task.goal,
                    "source_text": task.source_text,
                    "variables": list(task.variables),
                    "constraints": list(task.constraints),
                    "metrics": list(task.metrics),
                    "possible_model_types": list(task.possible_model_types),
                    "evidence": list(task.evidence),
                }
            )

        observed = list(profile.columns)
        decision = self._match_terms(
            observed,
            ("decision", "allocation", "schedule", "route", "选择", "分配", "调度", "路径", "方案"),
        )
        states = self._match_terms(
            observed,
            ("state", "stock", "inventory", "population", "状态", "库存", "存量", "人数"),
        )
        parameters = self._match_terms(
            observed,
            ("rate", "probability", "coefficient", "weight", "率", "概率", "系数", "权重"),
        )
        constraints = self._dedupe(
            [
                constraint
                for task in tasks
                for constraint in task.constraints
            ]
            + self._extract_sentences(text, ("不超过", "至少", "至多", "必须", "约束", "限制", "预算", "容量"))
        )
        metrics = self._dedupe(
            [metric for task in tasks for metric in task.metrics]
            + self._extract_metric_terms(text)
        )
        outputs = self._dedupe(
            [task.goal for task in tasks]
            + self._extract_sentences(text, ("输出", "给出", "求出", "预测", "评价", "排序", "方案"))
        )
        ambiguities = self._ambiguities(tasks, profile, decision)
        dependencies = self._infer_dependencies(subproblems)
        assumptions = self._initial_assumptions(tasks, profile)
        data_requirements = self._data_requirements(tasks, profile)

        return ProblemSpec(
            sub_questions=[task.source_text or task.goal for task in tasks],
            subproblems=subproblems,
            inputs=[str(path) for path in state.data_files],
            outputs=outputs,
            observed_variables=observed,
            decision_variables=decision,
            state_variables=states,
            parameters=parameters,
            constraints=constraints,
            assumptions=assumptions,
            metrics=metrics,
            time_scale=self._detect_scale(text, "time"),
            spatial_scale=self._detect_scale(text, "space"),
            uncertainty_sources=self._uncertainty_sources(text, profile),
            data_requirements=data_requirements,
            task_dependencies=dependencies,
            ambiguities=ambiguities,
            raw_analysis=state.notes.get("problem_analysis", ""),
        )

    def _format_analysis(self, spec: ProblemSpec, state: WorkflowState) -> str:
        data_summaries = summarize_data_files(state.data_files)
        lines = ["# 题目理解", "", "## 子问题"]
        for item in spec.subproblems:
            lines.append(
                f"- {item['id']} [{item['task_type']}]：{item['objective']}"
            )
        lines.extend(["", "## 数据附件"])
        lines.extend(
            f"- {summary.path.name}: {summary.file_type}, {summary.note}"
            for summary in data_summaries
        )
        if not data_summaries:
            lines.append("- 未提供结构化数据文件。")
        lines.extend(["", "## 变量与约束"])
        lines.append(f"- 观测变量：{', '.join(spec.observed_variables) or '待识别'}")
        lines.append(f"- 决策变量：{', '.join(spec.decision_variables) or '待定义'}")
        lines.append(f"- 约束：{'；'.join(spec.constraints) or '待提取'}")
        lines.extend(["", "## 数据与建模风险"])
        lines.extend(f"- {item}" for item in spec.ambiguities)
        if not spec.ambiguities:
            lines.append("- 暂未发现阻断性歧义，仍需在建模阶段验证假设。")
        return "\n".join(lines)

    def _data_columns(self, state: WorkflowState) -> list[str]:
        return list(DataProfileAgent().run(state.data_files).columns)

    def _extract_sentences(self, text: str, terms: tuple[str, ...]) -> list[str]:
        sentences = [s.strip() for s in re.split(r"[。！？；;\n]+", text) if s.strip()]
        return [s for s in sentences if any(term in s for term in terms)]

    def _extract_metric_terms(self, text: str) -> list[str]:
        aliases = {
            "RMSE": ("rmse", "均方根误差"),
            "MAE": ("mae", "平均绝对误差"),
            "MAPE": ("mape", "平均绝对百分比误差"),
            "R²": ("r2", "r²", "决定系数"),
            "准确率": ("准确率", "accuracy"),
            "F1": ("f1", "f1-score"),
            "目标函数值": ("目标函数", "最优值"),
        }
        lower = text.lower()
        return [name for name, terms in aliases.items() if any(term in lower for term in terms)]

    def _match_terms(self, columns: list[str], terms: tuple[str, ...]) -> list[str]:
        return [
            column
            for column in columns
            if any(term.lower() in column.lower() for term in terms)
        ]

    def _ambiguities(self, tasks, profile, decision_variables: list[str]) -> list[str]:
        issues: list[str] = []
        if not profile.has_data:
            issues.append("未提供可读取的数据，变量类型、样本量和模型条件无法验证。")
        if any(task.task_type == "forecast" for task in tasks) and not profile.datetime_columns:
            issues.append("预测任务未识别到明确时间字段或时间尺度。")
        if any(task.task_type == "classification" for task in tasks) and not profile.target_columns:
            issues.append("分类任务未识别到标签列。")
        if any(task.task_type == "optimization" for task in tasks) and not decision_variables:
            issues.append("优化任务尚未从数据字段中识别到明确决策变量。")
        if any(task.task_type == "evaluation" for task in tasks) and len(profile.numeric_columns) < 2:
            issues.append("综合评价任务缺少至少两个可计算的数值指标。")
        return issues

    def _infer_dependencies(self, subproblems: list[dict]) -> list[dict[str, str]]:
        dependencies: list[dict[str, str]] = []
        for index, current in enumerate(subproblems):
            current_type = current["task_type"]
            for previous in subproblems[:index]:
                previous_type = previous["task_type"]
                if (previous_type, current_type) in {
                    ("forecast", "optimization"),
                    ("evaluation", "optimization"),
                    ("statistics", "forecast"),
                    ("exploration", "classification"),
                    ("exploration", "clustering"),
                }:
                    dependencies.append(
                        {
                            "from": previous["id"],
                            "to": current["id"],
                            "reason": f"{current_type} 使用 {previous_type} 的结果或参数",
                        }
                    )
        return dependencies

    def _initial_assumptions(self, tasks, profile) -> list[str]:
        assumptions = ["输入数据的统计口径在各文件和各时间段之间保持一致。"]
        task_types = {task.task_type for task in tasks}
        if "forecast" in task_types:
            assumptions.append("历史规律在预测区间内具有一定延续性。")
        if "statistics" in task_types or "classification" in task_types:
            assumptions.append("样本具有足够代表性，训练与评估数据不存在信息泄漏。")
        if "optimization" in task_types:
            assumptions.append("目标函数和约束能够用给定变量近似表达。")
        if profile.missing_rate > 0:
            assumptions.append("缺失值处理不会系统性改变样本分布。")
        return assumptions

    def _data_requirements(self, tasks, profile) -> list[str]:
        requirements: list[str] = []
        task_types = {task.task_type for task in tasks}
        if "forecast" in task_types:
            requirements.append("连续、顺序明确的时间字段和目标序列。")
        if "classification" in task_types:
            requirements.append("明确的标签列及足够的各类别样本。")
        if "optimization" in task_types:
            requirements.append("目标系数、资源消耗、容量或预算约束。")
        if "network" in task_types:
            requirements.append("起点、终点以及可选的权重或容量字段。")
        if profile.rows:
            requirements.append(f"当前已读取约 {profile.rows} 行数据。")
        return requirements

    def _detect_scale(self, text: str, kind: str) -> str:
        terms = (
            ("小时", "天", "周", "月", "季度", "年")
            if kind == "time"
            else ("节点", "站点", "区域", "城市", "省", "全国")
        )
        return next((term for term in terms if term in text), "未明确")

    def _uncertainty_sources(self, text: str, profile) -> list[str]:
        sources: list[str] = []
        aliases = {
            "随机扰动": ("随机", "波动", "噪声"),
            "参数不确定性": ("参数不确定", "估计误差"),
            "需求不确定性": ("需求波动", "需求不确定"),
            "测量误差": ("测量误差", "观测误差"),
        }
        for label, terms in aliases.items():
            if any(term in text for term in terms):
                sources.append(label)
        if profile.missing_rate > 0:
            sources.append(f"缺失数据（总体缺失率约 {profile.missing_rate:.1%}）")
        return sources

    def _dedupe(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))
