from __future__ import annotations

import logging

from agents.base import (
    Agent,
    ExperimentPlan,
    K_MODELING_PLAN,
    K_RESULT_ANALYSIS,
    WorkflowState,
)
from tools.prompt_loader import load_prompt

log = logging.getLogger("mma.experiment_plan_agent")


class ExperimentPlanAgent(Agent):
    """Designs the experiment: metrics, data split, parameter grid, sensitivity.

    Runs after MODEL_DECISION.  Produces ``state.experiment_plan``.
    """

    name = "experiment_plan_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        plan = ExperimentPlan()

        if state.llm and state.llm.enabled:
            try:
                prompt = load_prompt("experiment_plan.md")
                response = state.llm.complete(prompt, self._build_llm_input(state))
                plan = self._parse_response(response, plan)
                state.experiment_plan = plan
                state.notes["experiment_plan_mode"] = "llm"
                return state
            except Exception as exc:
                state.notes["experiment_plan_llm_error"] = str(exc)
                log.warning("LLM experiment plan failed: %s", exc)

        # heuristic fallback
        plan = self._heuristic_plan(state)
        state.experiment_plan = plan
        state.notes["experiment_plan_mode"] = "heuristic"
        return state

    def _build_llm_input(self, state: WorkflowState) -> str:
        model_info = ""
        if state.model_decision:
            d = state.model_decision
            model_info = f"主模型: {d.primary_model_id}\n基线: {d.baseline_model_id}\n全部: {d.selected_model_ids}"

        return "\n\n".join([
            "题目：\n" + state.problem_text.strip()[:2000],
            "建模方案：\n" + state.notes.get(K_MODELING_PLAN, "")[:1000],
            "模型决策：\n" + model_info,
            "数据文件：\n" + "\n".join(f"- {p}" for p in state.data_files),
            "\n请设计实验方案，包含：",
            "1. 评估指标（metrics）",
            "2. 数据划分方式（训练/测试/验证）",
            "3. 参数网格（超参搜索范围）",
            "4. 敏感性分析方案",
            "5. 消融实验方案",
        ])

    def _parse_response(self, response: str, base: ExperimentPlan) -> ExperimentPlan:
        metrics: list[str] = []
        params: dict[str, list] = {}
        sensitivity = ""
        ablation = ""
        data_split = ""

        current_section = ""
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            lower = s.lower()

            if "指标" in s or "metric" in lower:
                current_section = "metrics"
                continue
            elif "划分" in s or "split" in lower:
                current_section = "split"
                continue
            elif "参数" in s or "grid" in lower or "param" in lower:
                current_section = "params"
                continue
            elif "敏感" in s or "sensitivity" in lower:
                current_section = "sensitivity"
                continue
            elif "消融" in s or "ablation" in lower:
                current_section = "ablation"
                continue

            if current_section == "metrics" and s.startswith("-"):
                metrics.append(s.lstrip("-* "))
            elif current_section == "split":
                data_split += s + "\n"
            elif current_section == "sensitivity":
                sensitivity += s + "\n"
            elif current_section == "ablation":
                ablation += s + "\n"

        base.metrics = metrics if metrics else base.metrics
        base.data_split = data_split.strip() if data_split else base.data_split
        base.validation_strategy = self._strategy_from_metrics(base.metrics)
        base.sensitivity_plan = sensitivity.strip() if sensitivity else base.sensitivity_plan
        base.ablation_plan = ablation.strip() if ablation else base.ablation_plan
        base.raw_plan = response
        return base

    def _heuristic_plan(self, state: WorkflowState) -> ExperimentPlan:
        plan = ExperimentPlan()

        # detect problem type to choose metrics
        problem_lower = state.problem_text.lower()
        if any(kw in problem_lower for kw in ("分类", "classif", "识别")):
            plan.metrics = ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]
            plan.validation_strategy = "stratified_k_fold"
        elif any(kw in problem_lower for kw in ("回归", "预测", "regress", "predict", "forecast")):
            plan.metrics = ["rmse", "mae", "r2_score", "mape"]
            plan.validation_strategy = "rolling_origin_backtest"
        elif any(kw in problem_lower for kw in ("聚类", "cluster", "分段")):
            plan.metrics = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score"]
            plan.validation_strategy = "cluster_stability_resampling"
        elif any(kw in problem_lower for kw in ("优化", "optim", "规划")):
            plan.metrics = ["objective_value", "constraint_violation", "runtime"]
            plan.validation_strategy = "feasibility_and_perturbation"
        else:
            plan.metrics = ["rmse", "mae", "r2_score"]
            plan.validation_strategy = "holdout_and_baseline_comparison"

        plan.data_split = "80% 训练 / 20% 测试（若数据量 > 1000 则采用 5 折交叉验证）"
        plan.test_size = 0.2
        plan.cv_folds = 5
        plan.random_seeds = [42, 2024, 2025]
        plan.parameter_grid = {}
        plan.sensitivity_plan = "对关键参数 ±20% 扰动，观察指标变化幅度"
        plan.ablation_plan = "逐一移除模型组件，评估每个组件对最终指标的贡献"
        plan.raw_plan = "启发式实验方案（未使用LLM）"
        return plan

    def _strategy_from_metrics(self, metrics: list[str]) -> str:
        lowered = " ".join(metrics).lower()
        if any(item in lowered for item in ("accuracy", "precision", "recall", "f1")):
            return "stratified_k_fold"
        if any(item in lowered for item in ("rmse", "mae", "mape", "r2")):
            return "rolling_origin_backtest"
        if "silhouette" in lowered or "davies" in lowered:
            return "cluster_stability_resampling"
        if "constraint" in lowered or "objective" in lowered:
            return "feasibility_and_perturbation"
        return "holdout_and_baseline_comparison"
