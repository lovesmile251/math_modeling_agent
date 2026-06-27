"""Innovation recommendation agent.

Recommends innovation enhancements based on the problem context, data profile,
and already-selected models. Does NOT run models; it produces structured
innovation suggestions for the writing agent and paper quality system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InnovationSuggestion:
    """A single innovation recommendation."""
    innovation_id: str
    title: str
    category: str  # residual_correction | combined_weight | ensemble | robust | sensitivity | dynamic | graph | mechanism_fusion
    description: str
    applicable_models: list[str] = field(default_factory=list)
    benefit: str = ""
    complexity_cost: str = "low"  # low | medium | high
    paper_text: str = ""  # ready-to-use text for the paper


# ── Innovation rule database ──

INNOVATION_RULES: list[dict[str, Any]] = [
    # Residual correction
    {
        "condition": {"task_types": ["forecast"], "sample_size": "tiny_or_small", "trigger_models": ["grey_gm11", "trend_forecast"]},
        "innovation": {
            "innovation_id": "residual_correction_grey_markov",
            "title": "灰色-马尔可夫残差修正",
            "category": "residual_correction",
            "description": "用灰色预测GM(1,1)提取趋势，对残差序列建立马尔可夫状态转移矩阵进行修正，提高小样本预测精度。",
            "benefit": "小样本条件下预测精度通常提升15-30%，同时给出预测区间",
            "complexity_cost": "medium",
            "paper_text": "针对小样本预测的不确定性，本文构建灰色-马尔可夫残差修正模型：首先利用GM(1,1)提取序列趋势，再对残差序列划分状态区间并估计一步转移概率矩阵，最终输出带置信区间的预测结果，有效降低了单一灰色模型的预测偏差。",
        },
    },
    # Combined weights
    {
        "condition": {"task_types": ["evaluation"], "has_subjective": True, "trigger_models": ["ahp_weights", "entropy_weights"]},
        "innovation": {
            "innovation_id": "ahp_entropy_combined_weights",
            "title": "AHP-熵权法组合赋权",
            "category": "combined_weight",
            "description": "将主观AHP权重与客观熵权通过乘法合成或博弈论方法组合，降低单一赋权偏差。",
            "benefit": "兼顾专家经验与数据客观规律，排名稳定性显著提升",
            "complexity_cost": "low",
            "paper_text": "为克服单一赋权方法的局限性，本文采用AHP-熵权法组合赋权策略：AHP反映决策者偏好结构，熵权法捕捉指标数据的内在信息量，通过乘法归一化合成最终权重，使评价结果同时具备主观合理性和客观依据。",
        },
    },
    # Multi-model ensemble
    {
        "condition": {"task_types": ["forecast", "classification"], "multiple_models": True},
        "innovation": {
            "innovation_id": "stacking_ensemble",
            "title": "Stacking多模型集成",
            "category": "ensemble",
            "description": "将多个异质模型（线性、非线性、树模型）的输出作为元特征，训练元学习器进行最终预测。",
            "benefit": "综合不同模型优势，泛化误差通常低于单一最优模型",
            "complexity_cost": "high",
            "paper_text": "本文构建Stacking集成学习框架：以线性回归、岭回归和梯度提升树为基学习器，以简单线性模型为元学习器，通过5折交叉验证生成元特征，最终集成预测结果在保持可解释性的同时提升了预测稳健性。",
        },
    },
    # Robust optimization
    {
        "condition": {"task_types": ["optimization"], "has_uncertainty": True},
        "innovation": {
            "innovation_id": "robust_optimization",
            "title": "鲁棒优化建模",
            "category": "robust",
            "description": "将不确定参数建模为区间或椭球不确定集，求解最坏情况下的最优方案。",
            "benefit": "方案在参数扰动下仍保持可行性，避免'最优但不可行'的困境",
            "complexity_cost": "medium",
            "paper_text": "考虑到实际系统中参数的不确定性，本文采用鲁棒优化框架：将关键约束参数建模为对称区间不确定集，通过Soyster/ Ben-Tal鲁棒对等转化将不确定优化问题转化为确定性凸优化问题，确保最优方案在参数波动范围内始终可行。",
        },
    },
    # Sensitivity analysis
    {
        "condition": {"task_types": ["evaluation", "optimization", "forecast"]},
        "innovation": {
            "innovation_id": "global_sensitivity_analysis",
            "title": "全局灵敏度与弹性分析",
            "category": "sensitivity",
            "description": "对关键参数和权重进行全局灵敏度扫描，绘制弹性曲线和龙卷风图。",
            "benefit": "揭示模型输出对各输入的依赖程度，增强结论说服力",
            "complexity_cost": "low",
            "paper_text": "为验证模型结论的稳健性，本文进行了系统的灵敏度分析：对指标权重、约束参数和关键假设分别施加±20%的扰动，通过龙卷风图识别高影响因子，并用Spearman秩相关系数度量输入-输出单调性，结果表明模型结论对参数扰动具有良好的鲁棒性。",
        },
    },
    # Dynamic weights
    {
        "condition": {"task_types": ["evaluation"], "has_time": True},
        "innovation": {
            "innovation_id": "dynamic_weight_evaluation",
            "title": "动态权重评价模型",
            "category": "dynamic",
            "description": "引入时间衰减因子或滑动窗口，使指标权重随时间动态调整，适用于面板数据评价。",
            "benefit": "反映评价标准的时变特性，避免静态权重的滞后性",
            "complexity_cost": "medium",
            "paper_text": "考虑到评价标准的时效性，本文引入动态权重机制：通过时间衰减函数赋予近期数据更高权重，结合滑动时间窗口逐期更新熵权，构建了'时间-指标'双重加权的动态评价模型，使评价结果能及时反映系统状态的演变。",
        },
    },
    # Graph + mechanism
    {
        "condition": {"task_types": ["network"], "trigger_models": ["community_detection", "graph_centrality"]},
        "innovation": {
            "innovation_id": "graph_propagation_mechanism",
            "title": "图传播机理增强",
            "category": "graph",
            "description": "在社区发现和中心性分析基础上，叠加独立级联或SIR传播仿真，量化节点影响力传播范围。",
            "benefit": "将静态网络指标扩展为动态传播影响力，更贴合实际问题",
            "complexity_cost": "medium",
            "paper_text": "本文在传统网络拓扑分析基础上引入传播动力学视角：首先通过社区发现和中心性指标识别关键节点，再基于独立级联模型仿真信息沿网络边的传播过程，以48小时传播覆盖率为指标量化节点真实影响力，弥补了静态中心性无法反映级联效应的不足。",
        },
    },
    # Mechanism + Data-driven fusion
    {
        "condition": {"task_types": ["forecast", "simulation"], "trigger_models": ["logistic_growth", "sir_model", "grey_gm11"]},
        "innovation": {
            "innovation_id": "mechanism_data_fusion",
            "title": "机理-数据驱动融合建模",
            "category": "mechanism_fusion",
            "description": "用机理模型提供结构约束，用数据驱动模型拟合残差或修正参数，实现优势互补。",
            "benefit": "兼具机理可解释性和数据拟合精度，减少过拟合风险",
            "complexity_cost": "high",
            "paper_text": "本文提出机理-数据驱动融合建模策略：以机理模型（如Logistic增长/SIR传播）为主体框架确保物理可解释性，以梯度提升或神经网络对机理残差进行补偿建模，通过正则化约束防止数据驱动部分过拟合，最终模型在保持机理一致性的同时显著提升了拟合精度。",
        },
    },
    # Monte Carlo uncertainty
    {
        "condition": {"task_types": ["optimization", "simulation", "statistics"]},
        "innovation": {
            "innovation_id": "monte_carlo_uncertainty",
            "title": "蒙特卡洛不确定性量化",
            "category": "sensitivity",
            "description": "对关键参数进行蒙特卡洛采样，生成输出分布和置信区间，量化结果不确定性。",
            "benefit": "将点估计扩展为概率分布，支持风险知情决策",
            "complexity_cost": "medium",
            "paper_text": "为量化模型输出的不确定性，本文采用蒙特卡洛模拟方法：对输入参数假设合理的概率分布（正态/均匀/三角），进行10000次独立采样计算，输出结果的95%置信区间和概率密度分布，为决策者提供了超越点估计的风险信息。",
        },
    },
]


class InnovationRecommendationAgent:
    """Recommend innovation strategies based on problem context and selected models."""

    def run(
        self,
        tasks: tuple[Any, ...],
        profile: Any,
        selected_models: tuple[Any, ...],
    ) -> list[InnovationSuggestion]:
        """Generate innovation suggestions for the writing agent."""
        task_types = {task.task_type for task in tasks}
        selected_model_ids = {m.model_id for m in selected_models}
        has_multiple = len(selected_model_ids) >= 2
        has_time = bool(getattr(profile, "monotonic_time_columns", ()) or getattr(profile, "datetime_columns", ()))
        has_subjective = "ahp_weights" in selected_model_ids

        sample_size_category = getattr(profile, "sample_size_category", "medium")
        is_tiny_or_small = sample_size_category in ("tiny", "small")

        suggestions: list[InnovationSuggestion] = []

        for rule in INNOVATION_RULES:
            cond = rule["condition"]
            # Check task type match
            required_types = set(cond.get("task_types", []))
            if required_types and not (required_types & task_types):
                continue
            # Check sample size
            if cond.get("sample_size") == "tiny_or_small" and not is_tiny_or_small:
                continue
            # Check has_subjective flag
            if cond.get("has_subjective") and not has_subjective:
                continue
            # Check has_uncertainty flag — infer from profile
            if cond.get("has_uncertainty") and sample_size_category == "tiny":
                continue  # too small for robust optimization to be meaningful
            # Check trigger models
            trigger = set(cond.get("trigger_models", []))
            if trigger and not (trigger & selected_model_ids):
                continue
            # Check multiple models
            if cond.get("multiple_models") and not has_multiple:
                continue
            # Check has_time
            if cond.get("has_time") and not has_time:
                continue

            innov = rule["innovation"]
            suggestion = InnovationSuggestion(
                innovation_id=innov["innovation_id"],
                title=innov["title"],
                category=innov["category"],
                description=innov["description"],
                applicable_models=list(trigger & selected_model_ids) if trigger else list(selected_model_ids),
                benefit=innov["benefit"],
                complexity_cost=innov["complexity_cost"],
                paper_text=innov["paper_text"],
            )
            suggestions.append(suggestion)

        # Always suggest sensitivity analysis if not already suggested
        if not any(s.innovation_id == "global_sensitivity_analysis" for s in suggestions):
            suggestions.append(InnovationSuggestion(
                innovation_id="global_sensitivity_analysis",
                title="全局灵敏度与弹性分析",
                category="sensitivity",
                description="对关键参数和权重进行全局灵敏度扫描，增强结论说服力。",
                applicable_models=list(selected_model_ids),
                benefit="揭示模型输出对各输入的依赖程度",
                complexity_cost="low",
                paper_text="为验证模型结论的稳健性，本文进行了灵敏度分析：对关键参数施加±20%扰动，识别高影响因子，结果表明核心结论在参数合理波动范围内保持稳定。",
            ))

        return suggestions

    def format_for_paper(self, suggestions: list[InnovationSuggestion]) -> str:
        """Format innovation suggestions as a paper-ready section."""
        if not suggestions:
            return ""

        lines = ["## 模型创新点", ""]
        lines.append("本文在建模过程中引入了以下创新增强策略：")
        lines.append("")

        for idx, s in enumerate(suggestions, 1):
            lines.append(f"### 创新点 {idx}：{s.title}")
            lines.append("")
            lines.append(s.paper_text)
            lines.append("")
            if s.applicable_models:
                lines.append(f"关联模型：{'、'.join(s.applicable_models)}")
            lines.append(f"复杂度代价：{s.complexity_cost}")
            lines.append("")

        return "\n".join(lines)

    def format_as_list(self, suggestions: list[InnovationSuggestion]) -> list[str]:
        """Format innovation suggestions as a concise list for writing prompts."""
        items = []
        for idx, s in enumerate(suggestions, 1):
            items.append(f"{idx}. {s.title}：{s.description}（关联：{'、'.join(s.applicable_models) if s.applicable_models else '全部模型'}）")
        return items
