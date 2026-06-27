from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from agents.base import (
    Agent,
    K_MODEL_SELECTION,
    K_PROBLEM_TYPE,
    K_SELECTED_MODEL_IDS,
    WorkflowState,
)
from agents.model_selection_crew import ModelSelectionCrew
from models.catalog import ALGORITHM_CATALOG, EXECUTABLE_MODEL_LABELS, AlgorithmEntry, catalog_size, executable_model_ids
from tools.file_tool import write_text


@dataclass(frozen=True)
class ModelChoice:
    model_id: str
    label: str
    reason: str
    source_problem: str


@dataclass(frozen=True)
class ReferenceChoice:
    category: str
    problem: str
    recommended: tuple[str, ...]
    alternatives: tuple[str, ...]
    reason: str


class ModelSelectionAgent(Agent):
    name = "model_selection_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        columns = self._read_columns(state)
        from models.social_network.campus import is_social_network_problem

        if is_social_network_problem(state.problem_text, columns):
            return self._run_social_network(state, columns)

        crew = ModelSelectionCrew(llm=state.llm)
        selection = crew.run(state.problem_text, state.data_files, columns)
        state.notes["model_selection_task_parser_mode"] = crew.task_agent.last_mode
        if crew.task_agent.last_error:
            state.notes["model_selection_task_parser_error"] = crew.task_agent.last_error
        executable_choices = selection.selected
        reference_choices = selection.references

        selected_ids = [choice.model_id for choice in executable_choices]
        state.notes[K_SELECTED_MODEL_IDS] = json.dumps(selected_ids, ensure_ascii=False)
        state.notes["selected_reference_algorithms"] = json.dumps(
            [
                {
                    "category": item.category,
                    "problem": item.problem,
                    "recommended": list(item.recommended),
                    "alternatives": list(item.alternatives),
                    "reason": item.reason,
                }
                for item in reference_choices
            ],
            ensure_ascii=False,
        )
        catalog_path = self._write_catalog(state)
        state.artifacts["algorithm_catalog"] = catalog_path
        report_path = write_text(state.workspace.logs_dir / "model_selection_report.json", selection.report_json)
        state.artifacts["model_selection_report"] = report_path
        state.notes[K_MODEL_SELECTION] = selection.report
        return state

    def _run_social_network(self, state: WorkflowState, columns: list[str]) -> WorkflowState:
        focused = [
            ("community_detection", "社群发现与高密度社群挖掘", "Louvain/贪心模块度 + 内部连接密度 + 社群间关系与重叠分析（问题一）"),
            ("friend_recommendation", "好友推荐（链路预测）", "共同邻居 / Jaccard / Adamic-Adar / 资源分配指数为目标用户推荐 Top-3（问题二）"),
            ("information_propagation", "信息传播仿真与关键用户筛选", "独立级联(IC)模型 + 多中心性候选 + 48 小时蒙特卡洛传播仿真（问题三）"),
            ("influence_maximization", "推送名额影响力最大化", "贪心边际增益最大化 + 与度/PageRank/随机基准对比（问题四）"),
        ]
        selected_ids = [model_id for model_id, _, _ in focused]
        state.notes[K_SELECTED_MODEL_IDS] = json.dumps(selected_ids, ensure_ascii=False)
        state.notes["selected_reference_algorithms"] = json.dumps([], ensure_ascii=False)
        state.notes[K_PROBLEM_TYPE] = "social_network"
        catalog_path = self._write_catalog(state)
        state.artifacts["algorithm_catalog"] = catalog_path
        lines = [
            "# 模型选择",
            "",
            "- 题型识别：社交网络分析（好友关系 / 社群 / 信息传播 / 推荐）。",
            "- 选型策略：针对题目四个子问题，仅启用解决问题的关键模型，不做无关模型堆砌。",
            "",
            "## 关键模型（按子问题）",
        ]
        for _, label, reason in focused:
            lines.append(f"- {label}：{reason}")
        lines.extend(
            [
                "",
                "## 说明",
                "- 上述模型全部在真实好友关系数据上运行，输出真实结果表与图表。",
                "- 行为数据表 / 用户属性表缺失时，相关参数（话题参与度、互动频率、活跃时段）采用基于网络结构的代理指标，并在结果中明确标注。",
            ]
        )
        state.notes[K_MODEL_SELECTION] = "\n".join(lines)
        return state

    def _match_catalog(self, problem_text: str, columns: list[str]) -> list[AlgorithmEntry]:
        text = f"{problem_text} {' '.join(columns)}".lower()
        scored: list[tuple[int, AlgorithmEntry]] = []
        for entry in ALGORITHM_CATALOG:
            score = 0
            terms = (
                entry.category,
                entry.problem,
                *entry.keywords,
                *entry.recommended,
                *entry.alternatives,
            )
            for term in terms:
                normalized = term.lower()
                if normalized and normalized in text:
                    score += 3 if term in entry.keywords else 1
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored]

    def _select_executable_models(self, matched_entries: list[AlgorithmEntry], columns: list[str]) -> list[ModelChoice]:
        valid_ids = executable_model_ids()
        choices: list[ModelChoice] = []
        for entry in matched_entries:
            for model_id in entry.executable_model_ids:
                if model_id not in valid_ids:
                    continue
                choices.append(
                    ModelChoice(
                        model_id=model_id,
                        label=EXECUTABLE_MODEL_LABELS[model_id],
                        reason=f"命中题型：{entry.category} / {entry.problem}",
                        source_problem=entry.problem,
                    )
                )

        catalog_text = " ".join(entry.problem for entry in matched_entries)
        if "需求" in catalog_text and "容量" in catalog_text:
            choices.append(
                ModelChoice(
                    "capacity_gap",
                    EXECUTABLE_MODEL_LABELS["capacity_gap"],
                    "命中需求与容量相关题型，补充供需缺口诊断。",
                    "需求容量约束诊断",
                )
            )

        if not choices and not matched_entries and columns:
            choices.extend(
                [
                    ModelChoice("trend_forecast", EXECUTABLE_MODEL_LABELS["trend_forecast"], "未命中明确题型，默认进行基础趋势探索。", "默认探索"),
                    ModelChoice("entropy_weights", EXECUTABLE_MODEL_LABELS["entropy_weights"], "未命中明确题型，默认计算数值指标客观权重。", "默认探索"),
                    ModelChoice("topsis_rank", EXECUTABLE_MODEL_LABELS["topsis_rank"], "未命中明确题型，默认生成综合评价排名。", "默认探索"),
                ]
            )

        deduped: dict[str, ModelChoice] = {}
        for choice in choices:
            deduped.setdefault(choice.model_id, choice)
        return list(deduped.values())

    def _select_reference_models(self, matched_entries: list[AlgorithmEntry]) -> list[ReferenceChoice]:
        references: list[ReferenceChoice] = []
        for entry in matched_entries:
            references.append(
                ReferenceChoice(
                    category=entry.category,
                    problem=entry.problem,
                    recommended=entry.recommended,
                    alternatives=entry.alternatives,
                    reason="命中题目关键词或数据字段。",
                )
            )

        deduped: dict[tuple[str, str], ReferenceChoice] = {}
        for item in references:
            deduped.setdefault((item.category, item.problem), item)
        return list(deduped.values())[:12]

    def _add_contextual_executable_choices(
        self,
        choices: list[ModelChoice],
        problem_text: str,
        columns: list[str],
    ) -> list[ModelChoice]:
        text = f"{problem_text} {' '.join(columns)}".lower()
        classification_terms = ("二分类", "多分类", "分类", "label", "class", "逻辑回归", "朴素贝叶斯", "knn")
        linear_regression_terms = ("线性回归", "多元回归", "linear regression", "ols")
        if any(term in text for term in classification_terms) and not any(term in text for term in linear_regression_terms):
            choices = [choice for choice in choices if choice.model_id != "linear_regression"]

        if "ahp" in text or "层次分析" in text or "层次分析法" in text:
            choices = [choice for choice in choices if choice.model_id not in {"kmeans_cluster"}]
            choices.append(
                ModelChoice(
                    "ahp_weights",
                    EXECUTABLE_MODEL_LABELS["ahp_weights"],
                    "题目明确包含 AHP 或层次分析法。",
                    "多指标综合评价（主观）",
                )
            )
        if any(term in text for term in ("需求", "demand", "need")) and any(term in text for term in ("容量", "capacity", "supply")):
            choices.append(
                ModelChoice(
                    "capacity_gap",
                    EXECUTABLE_MODEL_LABELS["capacity_gap"],
                    "题目或字段同时包含需求与容量，补充供需缺口诊断。",
                    "需求容量约束诊断",
                )
            )
        deduped: dict[str, ModelChoice] = {}
        for choice in choices:
            deduped.setdefault(choice.model_id, choice)
        return list(deduped.values())

    def _read_columns(self, state: WorkflowState) -> list[str]:
        columns: list[str] = []
        for path in state.data_files:
            try:
                if path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path, nrows=5)
                elif path.suffix.lower() == ".tsv":
                    frame = pd.read_csv(path, sep="\t", nrows=5)
                elif path.suffix.lower() in {".xlsx", ".xls"}:
                    frame = pd.read_excel(path, nrows=5)
                else:
                    continue
                columns.extend(str(column) for column in frame.columns)
            except Exception:
                continue
        return columns

    def _format_report(self, executable_choices: list[ModelChoice], reference_choices: list[ReferenceChoice]) -> str:
        lines = [
            "# 模型选择",
            "",
            f"- 已接入算法目录条目数：{catalog_size()}",
            f"- 当前可直接运行模型数：{len(EXECUTABLE_MODEL_LABELS)}",
            "",
            "## 可执行模型",
        ]
        if executable_choices:
            for choice in executable_choices:
                lines.append(f"- {choice.label}（`{choice.model_id}`）：{choice.reason}")
        else:
            lines.append("- 未选择可执行模型。")

        lines.extend(["", "## 参考算法推荐"])
        if reference_choices:
            for item in reference_choices:
                recommended = "、".join(item.recommended)
                alternatives = "、".join(item.alternatives) if item.alternatives else "无"
                lines.append(f"- {item.category} / {item.problem}：推荐 {recommended}；备选 {alternatives}")
        else:
            lines.append("- 未命中算法目录中的具体题型，可补充题目关键词或数据说明。")

        lines.extend(
            [
                "",
                "## 说明",
                "- 可执行模型会进入自动代码生成和结果表输出。",
                "- 参考算法已接入算法目录，可用于建模方案推荐；若需要自动运行，需要继续实现对应求解器。",
            ]
        )
        return "\n".join(lines)

    def _write_catalog(self, state: WorkflowState):
        payload = [
            {
                "category": item.category,
                "problem": item.problem,
                "recommended": list(item.recommended),
                "alternatives": list(item.alternatives),
                "keywords": list(item.keywords),
                "executable_model_ids": list(item.executable_model_ids),
            }
            for item in ALGORITHM_CATALOG
        ]
        return write_text(
            state.workspace.logs_dir / "algorithm_catalog.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
