from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from models.catalog import (
    ALGORITHM_CATALOG,
    EXECUTABLE_MODEL_LABELS,
    AlgorithmEntry,
    catalog_size,
    check_applicable,
    executable_model_ids,
    get_model_contract,
)
from agents.innovation_agent import InnovationRecommendationAgent
from tools.llm_client import parse_json_object
from tools.prompt_loader import load_prompt


@dataclass(frozen=True)
class SelectionTask:
    task_id: str
    task_type: str
    goal: str
    evidence: tuple[str, ...] = ()
    source_text: str = ""
    variables: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    possible_model_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataProfile:
    columns: tuple[str, ...]
    rows: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    id_like_columns: tuple[str, ...]
    demand_columns: tuple[str, ...]
    capacity_columns: tuple[str, ...]
    cost_columns: tuple[str, ...]
    benefit_columns: tuple[str, ...]
    relation_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    constraint_columns: tuple[str, ...]
    source_files: tuple[str, ...]
    unique_value_ratios: dict[str, float] = field(default_factory=dict)
    monotonic_time_columns: tuple[str, ...] = ()
    binary_label_columns: tuple[str, ...] = ()
    multiclass_label_columns: tuple[str, ...] = ()
    source_target_pairs: tuple[tuple[str, str], ...] = ()
    demand_capacity_pairs: tuple[tuple[str, str], ...] = ()
    resource_columns: tuple[str, ...] = ()
    objective_constraint_columns: tuple[str, ...] = ()
    has_edge_table: bool = False
    has_objective_constraint_combo: bool = False
    nonlinearity_score: float = 0.0  # 0=linear, 1=highly nonlinear
    missing_rate: float = 0.0  # overall missing value proportion
    sample_size_category: str = "unknown"  # tiny | small | medium | large | big

    @property
    def has_data(self) -> bool:
        return bool(self.columns)


@dataclass
class CandidateModel:
    model_id: str
    label: str
    task_type: str
    semantic_score: int = 0
    data_score: int = 0
    task_score: int = 0
    interpretability_score: int = 6
    risk_penalty: int = 0
    role: str = "candidate"
    task_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    source_problem: str = ""
    applicability: dict[str, Any] = field(default_factory=dict)
    tier: str = "baseline"  # baseline | improved | innovation
    explainability_score: int = 6  # 0-10: how easy to explain the model
    innovation_score: int = 0  # 0-10: novelty of the modeling approach
    applicability_score: int = 0  # 0-10: fit to data conditions
    validation_plan: list[str] = field(default_factory=list)  # suggested validation methods
    comparison_candidates: list[str] = field(default_factory=list)  # models to compare against
    contract: dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> int:
        return (
            self.semantic_score
            + self.data_score
            + self.task_score
            + self.interpretability_score
            + self.explainability_score
            + self.innovation_score
            + self.applicability_score
            - self.risk_penalty
        )


@dataclass(frozen=True)
class ReferenceChoice:
    category: str
    problem: str
    recommended: tuple[str, ...]
    alternatives: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ModelSelectionResult:
    tasks: tuple[SelectionTask, ...]
    data_profile: DataProfile
    selected: tuple[CandidateModel, ...]
    rejected: tuple[CandidateModel, ...]
    references: tuple[ReferenceChoice, ...]
    report: str
    report_json: str
    innovation_suggestions: list[Any] = field(default_factory=list)


class TaskDecompositionAgent:
    """Identify modeling subtasks from the problem text and data columns."""

    ALLOWED_TASK_TYPES = {
        "forecast",
        "evaluation",
        "optimization",
        "classification",
        "clustering",
        "network",
        "statistics",
        "simulation",
        "exploration",
    }

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm
        self.last_mode = "rule"
        self.last_error = ""

    KEYWORDS: dict[str, tuple[str, ...]] = {
        "forecast": (
            "预测", "趋势", "未来", "时间序列", "外推", "forecast", "trend",
            "future", "time series",
        ),
        "evaluation": (
            "评价", "评估", "综合评价", "指标体系", "权重", "排序", "排名", "缺口",
            "score", "rank", "weight", "evaluate", "assessment", "gap", "shortage",
        ),
        "optimization": (
            "优化", "分配", "调度", "路径", "约束", "成本", "收益", "最优",
            "决策", "策略", "配置", "排班", "安排", "方案", "最大化", "最小化",
            "尽可能大", "尽可能小", "测线", "覆盖", "重叠率", "时点选择",
            "optimize", "optimization", "schedule", "allocation", "constraint", "budget", "maximize",
            "minimize", "assignment", "knapsack",
        ),
        "classification": (
            "分类", "标签", "类别", "识别", "鉴别", "判定", "标注",
            "label", "class", "classifier", "classification",
        ),
        "clustering": (
            "聚类", "分群", "社群", "社区", "簇", "cluster", "segment",
            "clustering", "community", "communities",
        ),
        "network": (
            "网络", "图论", "最短路", "最大流", "社区", "好友", "交通网络",
            "道路网络", "路网", "交叉口", "服务平台",
            "network", "graph", "shortest path", "maximum flow", "max flow",
            "community", "edge", "node",
        ),
        "statistics": (
            "相关性", "相关关系", "相关分析", "回归", "假设检验", "方差",
            "参数估计", "因素分析",
            "拟合", "校准", "标定", "估计", "显著性", "误差分析",
            "关系", "测量", "反演", "可靠性", "抽样", "置信度", "信度",
            "correlation",
            "regression", "hypothesis", "anova", "statistical", "estimation",
            "pca", "dimensionality reduction", "主成分", "降维",
        ),
        "simulation": (
            "仿真", "模拟", "蒙特卡洛", "传播", "传染", "动力学", "机理模型",
            "运动模型", "微分方程", "概率模型", "命中概率", "动态过程",
            "热传导", "传热", "温度分布", "压力变化", "漫延", "流动过程",
            "simulation", "monte carlo", "epidemic", "dynamics",
        ),
    }

    GOALS: dict[str, str] = {
        "forecast": "预测未来趋势或数值",
        "evaluation": "构建指标体系并形成综合评价",
        "optimization": "在约束下给出最优或近似最优方案",
        "classification": "根据特征识别类别或标签",
        "clustering": "发现样本群组或结构",
        "network": "分析节点、边和网络结构",
        "statistics": "解释变量关系并进行统计检验",
        "simulation": "用仿真或机理模型刻画动态过程",
        "exploration": "先进行数据探索和基础建模",
    }

    def run(self, problem_text: str, columns: list[str]) -> tuple[SelectionTask, ...]:
        self.last_mode = "rule"
        self.last_error = ""
        if self.llm and self.llm.enabled:
            try:
                tasks = self._run_llm(problem_text, columns)
                if tasks:
                    self.last_mode = "llm"
                    return tasks
                self.last_error = "LLM task decomposition returned no tasks."
            except Exception as exc:
                self.last_mode = "fallback"
                self.last_error = str(exc)
        return self._run_rules(problem_text, columns)

    def _run_rules(self, problem_text: str, columns: list[str]) -> tuple[SelectionTask, ...]:
        segments = self._segments(problem_text)
        column_text = " ".join(columns)
        tasks: list[SelectionTask] = []
        seen: set[tuple[str, str]] = set()
        for segment_id, segment in segments:
            text = f"{segment} {column_text}".lower()
            for task_type, terms in self.KEYWORDS.items():
                hits = tuple(term for term in terms if self._term_matches(term, text))
                if not hits:
                    continue
                if (
                    task_type == "classification"
                    and set(hits) == {"分类"}
                    and "分类指标" in text
                ):
                    continue
                if (
                    task_type == "evaluation"
                    and any(term in text for term in ("network", "graph", "community", "网络", "社区"))
                    and set(hits).issubset({"rank", "排名", "排序"})
                ):
                    continue
                key = (segment_id, task_type)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    SelectionTask(
                        task_id=segment_id if len(segments) > 1 else f"T{len(tasks) + 1}",
                        task_type=task_type,
                        goal=self.GOALS[task_type],
                        evidence=hits[:6],
                        source_text=segment[:160],
                    )
                )
        if not tasks:
            tasks.append(SelectionTask("T1", "exploration", self.GOALS["exploration"], (), problem_text[:160]))
        return tuple(tasks)

    def _term_matches(self, term: str, text: str) -> bool:
        normalized = term.lower()
        if normalized.isascii() and any(char.isalnum() for char in normalized):
            pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
            return re.search(pattern, text) is not None
        return normalized in text

    def _run_llm(self, problem_text: str, columns: list[str]) -> tuple[SelectionTask, ...]:
        prompt = load_prompt("task_decomposition_json.md")
        llm_input = self._build_llm_input(problem_text, columns)
        complete_json = getattr(self.llm, "complete_json", None)
        if callable(complete_json):
            payload = complete_json(prompt, llm_input, schema={"type": "object"})
        else:
            response = self.llm.complete(prompt, llm_input)
            payload = self._parse_json_response(response)
        raw_tasks = payload.get("subproblems") or payload.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("LLM response must contain a subproblems array.")

        tasks: list[SelectionTask] = []
        for index, item in enumerate(raw_tasks, start=1):
            if not isinstance(item, dict):
                continue
            task_type = self._normalize_task_type(item.get("task_type") or item.get("type"))
            task_id = str(item.get("id") or item.get("task_id") or f"T{index}").strip() or f"T{index}"
            goal = str(item.get("objective") or item.get("goal") or self.GOALS[task_type]).strip()
            source_text = str(item.get("source_text") or item.get("description") or "").strip()
            tasks.append(
                SelectionTask(
                    task_id=task_id,
                    task_type=task_type,
                    goal=goal or self.GOALS[task_type],
                    evidence=self._string_tuple(item.get("evidence"))[:6],
                    source_text=source_text[:160],
                    variables=self._string_tuple(item.get("variables")),
                    constraints=self._string_tuple(item.get("constraints")),
                    metrics=self._string_tuple(item.get("metrics") or item.get("evaluation_metrics")),
                    possible_model_types=self._string_tuple(
                        item.get("possible_model_types") or item.get("model_types") or item.get("models")
                    ),
                )
            )
        return tuple(tasks)

    def _build_llm_input(self, problem_text: str, columns: list[str]) -> str:
        column_lines = "\n".join(f"- {column}" for column in columns) or "- none"
        return "\n".join(
            [
                "Problem text:",
                problem_text,
                "",
                "Data columns:",
                column_lines,
                "",
                "Allowed task_type values:",
                ", ".join(sorted(self.ALLOWED_TASK_TYPES)),
            ]
        )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        return parse_json_object(text)

    def _normalize_task_type(self, value: Any) -> str:
        task_type = str(value or "").strip().lower()
        if task_type in self.ALLOWED_TASK_TYPES:
            return task_type
        return "exploration"

    def _string_tuple(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list | tuple):
            items = value
        else:
            items = [value]
        return tuple(str(item).strip() for item in items if str(item).strip())

    def _segments(self, problem_text: str) -> list[tuple[str, str]]:
        text = problem_text.strip()
        if not text:
            return [("T1", "")]
        explicit_pattern = re.compile(
            r"(?:^|\n|[。；;])\s*(?:"
            r"问题\s*([一二三四五六七八九十\d]+)[.．、：:]?"
            r"|第\s*([一二三四五六七八九十\d]+)\s*问题"
            r"|Q\s*(\d+)[.．、：:]?"
            r")",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = list(explicit_pattern.finditer(text))
        if len(matches) < 2:
            fallback_pattern = re.compile(
                r"^\s*(?:[（(]\s*(\d+)\s*[）)]|(\d+)[.、]\s+)",
                re.MULTILINE,
            )
            matches = list(fallback_pattern.finditer(text))
        if len(matches) < 2:
            return [("T1", text)]

        segments: list[tuple[str, str]] = []
        for idx, match in enumerate(matches):
            start = 0 if idx == 0 else match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            raw_id = next((group for group in match.groups() if group), str(idx + 1))
            task_id = f"Q{self._normalize_number(raw_id)}"
            segment = text[start:end].strip()
            if segment:
                segments.append((task_id, segment))
        return segments or [("T1", text)]

    def _normalize_number(self, raw: str) -> str:
        digits = {
            "一": "1",
            "二": "2",
            "三": "3",
            "四": "4",
            "五": "5",
            "六": "6",
            "七": "7",
            "八": "8",
            "九": "9",
            "十": "10",
        }
        return digits.get(raw, raw)


class DataProfileAgent:
    """Build a lightweight tabular data profile for model applicability checks."""

    DEMAND_TERMS = ("demand", "need", "需求", "客流", "销量")
    CAPACITY_TERMS = ("capacity", "supply", "limit", "容量", "供给", "产能", "上限")
    COST_TERMS = ("cost", "price", "expense", "成本", "费用", "价格")
    BENEFIT_TERMS = ("profit", "benefit", "revenue", "score", "收益", "收入", "效益", "评分")
    TARGET_TERMS = ("target", "label", "class", "result", "outcome", "目标", "标签", "类别", "结果")
    CONSTRAINT_TERMS = ("constraint", "limit", "bound", "budget", "resource", "约束", "限制", "预算", "资源")
    ID_TERMS = ("id", "编号", "用户", "对象", "name")
    RELATION_TERMS = ("source", "target", "from", "to", "edge", "friend", "user1", "user2", "起点", "终点", "好友")
    TIME_TERMS = ("time", "date", "year", "month", "day", "时间", "日期", "年份", "月份")

    def run(self, data_files: list[Path]) -> DataProfile:
        frames: list[pd.DataFrame] = []
        source_files: list[str] = []
        for path in data_files:
            frame = self._read_table(path)
            if frame is None:
                continue
            frames.append(frame)
            source_files.append(str(path))

        if not frames:
            return DataProfile((), 0, (), (), (), (), (), (), (), (), (), (), (), tuple(source_files))

        columns = tuple(dict.fromkeys(str(column) for frame in frames for column in frame.columns))
        rows = sum(int(frame.shape[0]) for frame in frames)
        sample = pd.concat([frame.head(200) for frame in frames], ignore_index=True, sort=False)
        numeric = tuple(str(column) for column in sample.select_dtypes(include="number").columns)
        categorical = tuple(str(column) for column in sample.select_dtypes(exclude="number").columns)
        unique_ratios = self._unique_value_ratios(columns, sample)
        datetime_cols = tuple(column for column in columns if self._is_datetime_like(column, sample))
        monotonic_time = tuple(column for column in datetime_cols if self._is_monotonic_time_column(column, sample))
        id_like = tuple(column for column in columns if self._has_any(column, self.ID_TERMS))
        demand = tuple(column for column in columns if self._has_any(column, self.DEMAND_TERMS))
        capacity = tuple(column for column in columns if self._has_any(column, self.CAPACITY_TERMS))
        cost = tuple(column for column in columns if self._has_any(column, self.COST_TERMS))
        benefit = tuple(column for column in columns if self._has_any(column, self.BENEFIT_TERMS))
        relation = tuple(column for column in columns if self._has_any(column, self.RELATION_TERMS))
        binary_labels, multiclass_labels = self._label_columns(columns, sample, unique_ratios)
        target = tuple(
            dict.fromkeys(
                [
                    *(column for column in columns if self._has_any(column, self.TARGET_TERMS)),
                    *binary_labels,
                    *multiclass_labels,
                ]
            )
        )
        constraint = tuple(column for column in columns if self._has_any(column, self.CONSTRAINT_TERMS))
        resource = tuple(column for column in columns if self._is_resource_like(column))
        source_target_pairs = self._source_target_pairs(columns, sample, unique_ratios)
        if source_target_pairs:
            relation = tuple(dict.fromkeys([*relation, *source_target_pairs[0]]))
        demand_capacity_pairs = self._paired_columns(demand, capacity)
        objective_constraint = tuple(dict.fromkeys([*cost, *benefit, *constraint, *resource, *capacity]))
        has_objective_constraint_combo = bool((cost or benefit) and (constraint or resource or capacity))

        # Compute new profile fields
        nonlinearity_score = self._estimate_nonlinearity(sample, numeric)
        missing_rate = self._compute_missing_rate(sample, columns)
        sample_size_category = self._categorize_sample_size(rows)

        return DataProfile(
            columns=columns,
            rows=rows,
            numeric_columns=numeric,
            categorical_columns=categorical,
            datetime_columns=datetime_cols,
            id_like_columns=id_like,
            demand_columns=demand,
            capacity_columns=capacity,
            cost_columns=cost,
            benefit_columns=benefit,
            relation_columns=relation,
            target_columns=target,
            constraint_columns=constraint,
            source_files=tuple(source_files),
            unique_value_ratios=unique_ratios,
            monotonic_time_columns=monotonic_time,
            binary_label_columns=binary_labels,
            multiclass_label_columns=multiclass_labels,
            source_target_pairs=source_target_pairs,
            demand_capacity_pairs=demand_capacity_pairs,
            resource_columns=resource,
            objective_constraint_columns=objective_constraint,
            has_edge_table=bool(source_target_pairs),
            has_objective_constraint_combo=has_objective_constraint_combo,
            nonlinearity_score=nonlinearity_score,
            missing_rate=missing_rate,
            sample_size_category=sample_size_category,
        )

    def _read_table(self, path: Path) -> pd.DataFrame | None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return self._read_csv(path)
            if suffix == ".tsv":
                return pd.read_csv(path, sep="\t", encoding="utf-8-sig", nrows=1000)
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path, nrows=1000)
        except Exception:
            return None
        return None

    def _read_csv(self, path: Path) -> pd.DataFrame:
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=encoding, nrows=1000)
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                return pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    nrows=1000,
                )
        return pd.read_csv(path, sep=None, engine="python", nrows=1000)

    def _is_datetime_like(self, column: str, sample: pd.DataFrame) -> bool:
        if self._has_any(column, self.TIME_TERMS) or self._has_any(column, ("period", "step", "season", "quarter", "cycle", "phase")):
            return True
        if column not in sample.columns:
            return False
        series = sample[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            return False
        preview = series.astype(str).head(20)
        if not preview.str.contains(r"\d{2,4}[-/年]|\d{1,2}月|\d{1,2}日", regex=True).any():
            return False
        try:
            parsed = pd.to_datetime(preview, errors="coerce")
        except Exception:
            return False
        return bool(parsed.notna().mean() >= 0.7)

    def _unique_value_ratios(self, columns: tuple[str, ...], sample: pd.DataFrame) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for column in columns:
            if column not in sample.columns:
                continue
            series = sample[column].dropna()
            ratios[column] = 0.0 if series.empty else round(float(series.nunique(dropna=True) / len(series)), 4)
        return ratios

    def _is_monotonic_time_column(self, column: str, sample: pd.DataFrame) -> bool:
        if column not in sample.columns:
            return False
        series = sample[column].dropna()
        if len(series) < 3:
            return False
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
        else:
            values = pd.to_datetime(series, errors="coerce")
        values = values.dropna()
        if len(values) < 3:
            return False
        return bool(values.is_monotonic_increasing or values.is_monotonic_decreasing)

    def _label_columns(
        self,
        columns: tuple[str, ...],
        sample: pd.DataFrame,
        unique_ratios: dict[str, float],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        binary: list[str] = []
        multiclass: list[str] = []
        row_count = max(1, len(sample))
        for column in columns:
            if column not in sample.columns:
                continue
            series = sample[column].dropna()
            if len(series) < 4:
                continue
            unique_count = int(series.nunique(dropna=True))
            if unique_count < 2:
                continue
            name_hit = self._has_any(column, self.TARGET_TERMS)
            ratio = unique_ratios.get(column, 1.0)
            low_cardinality = ratio <= 0.35 or unique_count <= max(3, int(row_count * 0.2))
            if unique_count == 2 and (name_hit or low_cardinality):
                binary.append(column)
            elif 2 < unique_count <= 20 and (name_hit or low_cardinality):
                multiclass.append(column)
        return tuple(binary), tuple(multiclass)

    def _source_target_pairs(
        self,
        columns: tuple[str, ...],
        sample: pd.DataFrame,
        unique_ratios: dict[str, float],
    ) -> tuple[tuple[str, str], ...]:
        explicit_sources = [column for column in columns if self._has_any(column, ("source", "from", "origin", "start", "src"))]
        explicit_targets = [column for column in columns if self._has_any(column, ("target", "to", "dest", "end", "dst"))]
        pairs = [(source, target) for source in explicit_sources for target in explicit_targets if source != target]
        if pairs:
            return tuple(dict.fromkeys(pairs))

        candidates: list[str] = []
        for column in columns:
            if column not in sample.columns:
                continue
            series = sample[column].dropna()
            if len(series) < 2:
                continue
            unique_count = int(series.nunique(dropna=True))
            if unique_count >= 2 and unique_ratios.get(column, 1.0) <= 0.95:
                candidates.append(column)
        if len(candidates) < 2:
            return ()

        best_pair = max(
            ((left, right) for index, left in enumerate(candidates) for right in candidates[index + 1 :]),
            key=lambda pair: self._overlap_score(sample[pair[0]], sample[pair[1]]),
        )
        if self._overlap_score(sample[best_pair[0]], sample[best_pair[1]]) <= 0:
            return ()
        return (best_pair,)

    def _overlap_score(self, left: pd.Series, right: pd.Series) -> int:
        left_values = set(left.dropna().astype(str))
        right_values = set(right.dropna().astype(str))
        return len(left_values & right_values)

    def _paired_columns(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        return tuple((left_column, right_column) for left_column in left for right_column in right)

    def _is_resource_like(self, column: str) -> bool:
        return self._has_any(column, ("resource", "stock", "labor", "worker", "machine", "material", "crew", "asset"))

    def _has_any(self, value: str, terms: tuple[str, ...]) -> bool:
        normalized = str(value).lower()
        return any(term.lower() in normalized for term in terms)

    def _estimate_nonlinearity(self, sample: pd.DataFrame, numeric_columns: tuple[str, ...]) -> float:
        """Estimate nonlinearity 0 (linear) to 1 (highly nonlinear) via R² of linear fit."""
        if len(numeric_columns) < 2:
            return 0.0
        import numpy as np
        pool = [c for c in numeric_columns if c in sample.columns]
        if len(pool) < 2:
            return 0.0
        try:
            # Pick the last numeric column as candidate target
            target_col = pool[-1]
            feature_cols = [c for c in pool[:-1] if c != target_col]
            if not feature_cols:
                return 0.0
            sub = sample[[target_col] + feature_cols].dropna()
            if len(sub) < 4:
                return 0.0
            y = sub[target_col].to_numpy(dtype=float)
            X = sub[feature_cols].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(X)), X])
            coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
            predicted = design @ coeffs
            ss_total = float(np.sum((y - y.mean()) ** 2))
            if ss_total == 0:
                return 0.0
            r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / ss_total
            return round(max(0.0, min(1.0, 1.0 - r2)), 3)
        except Exception:
            return 0.5

    def _compute_missing_rate(self, sample: pd.DataFrame, columns: tuple[str, ...]) -> float:
        """Compute overall missing value proportion across all columns."""
        if sample.empty or not columns:
            return 0.0
        pool = [c for c in columns if c in sample.columns]
        if not pool:
            return 0.0
        total = sample[pool].size
        missing = sample[pool].isnull().sum().sum()
        return round(float(missing / total), 4) if total > 0 else 0.0

    def _categorize_sample_size(self, rows: int) -> str:
        if rows == 0:
            return "unknown"
        if rows < 10:
            return "tiny"
        if rows < 50:
            return "small"
        if rows < 500:
            return "medium"
        if rows < 5000:
            return "large"
        return "big"


class CandidateModelAgent:
    """Generate model candidates from catalog hits and task-level defaults."""

    TASK_MODEL_HINTS: dict[str, tuple[str, ...]] = {
        "forecast": ("trend_forecast", "smoothing_forecast", "grey_gm11", "ridge_regression"),
        "evaluation": ("entropy_weights", "topsis_rank", "grey_relation", "vikor"),
        "optimization": ("resource_allocation", "knapsack_01", "assignment_plan", "scheduling_plan"),
        "classification": ("logistic_classifier", "naive_bayes_classifier", "knn_classifier"),
        "clustering": ("kmeans_cluster", "dbscan_cluster", "hierarchical_cluster", "pca"),
        "network": ("graph_centrality", "graph_shortest_paths", "graph_mst", "community_detection"),
        "statistics": ("correlation_analysis", "linear_regression", "parameter_estimation", "hypothesis_tests", "quality_sampling_plan"),
        "simulation": ("monte_carlo", "logistic_growth", "sir_model", "signal_denoising"),
        "exploration": ("trend_forecast", "entropy_weights", "topsis_rank"),
    }

    def run(
        self,
        problem_text: str,
        profile: DataProfile,
        tasks: tuple[SelectionTask, ...],
    ) -> tuple[CandidateModel, ...]:
        matched_entries = self._match_catalog(problem_text, list(profile.columns))
        candidates: dict[str, CandidateModel] = {}

        for score, entry in matched_entries:
            for model_id in entry.executable_model_ids:
                self._add_candidate(
                    candidates,
                    model_id=model_id,
                    task_type=self._infer_task_type(entry),
                    semantic_score=min(40, score * 6),
                    reason=f"命中算法目录：{entry.category} / {entry.problem}",
                    source_problem=entry.problem,
                    task_id="catalog",
                )

        for task in tasks:
            for model_id in self.TASK_MODEL_HINTS.get(task.task_type, ()):
                self._add_candidate(
                    candidates,
                    model_id=model_id,
                    task_type=task.task_type,
                    semantic_score=18 if task.task_type != "exploration" else 10,
                    reason=f"服务任务 {task.task_id}：{task.goal}",
                    source_problem=task.goal,
                    task_id=task.task_id,
                )

        self._add_contextual_candidates(candidates, problem_text, profile)
        return tuple(candidates.values())

    def references(self, problem_text: str, columns: list[str]) -> tuple[ReferenceChoice, ...]:
        references: list[ReferenceChoice] = []
        for _, entry in self._match_catalog(problem_text, columns):
            references.append(
                ReferenceChoice(
                    category=entry.category,
                    problem=entry.problem,
                    recommended=entry.recommended,
                    alternatives=entry.alternatives,
                    reason="命中题目关键词或数据字段",
                )
            )
        deduped: dict[tuple[str, str], ReferenceChoice] = {}
        for item in references:
            deduped.setdefault((item.category, item.problem), item)
        return tuple(deduped.values())[:12]

    def _match_catalog(self, problem_text: str, columns: list[str]) -> list[tuple[int, AlgorithmEntry]]:
        text = f"{problem_text} {' '.join(columns)}".lower()
        scored: list[tuple[int, AlgorithmEntry]] = []
        for entry in ALGORITHM_CATALOG:
            score = 0
            for term in (entry.category, entry.problem, *entry.keywords, *entry.recommended, *entry.alternatives):
                normalized = term.lower()
                if normalized and normalized in text:
                    score += 3 if term in entry.keywords else 1
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _add_candidate(
        self,
        candidates: dict[str, CandidateModel],
        model_id: str,
        task_type: str,
        semantic_score: int,
        reason: str,
        source_problem: str,
        task_id: str = "",
    ) -> None:
        if model_id not in executable_model_ids():
            return
        if model_id not in candidates:
            candidates[model_id] = CandidateModel(
                model_id=model_id,
                label=EXECUTABLE_MODEL_LABELS[model_id],
                task_type=task_type,
                semantic_score=semantic_score,
                source_problem=source_problem,
                task_ids=[task_id] if task_id else [],
                reasons=[reason],
            )
            return
        item = candidates[model_id]
        item.semantic_score = min(100, item.semantic_score + semantic_score // 2)
        item.reasons.append(reason)
        if task_id and task_id not in item.task_ids:
            item.task_ids.append(task_id)
        if item.task_type == "exploration" and task_type != "exploration":
            item.task_type = task_type

    def _add_contextual_candidates(
        self,
        candidates: dict[str, CandidateModel],
        problem_text: str,
        profile: DataProfile,
    ) -> None:
        text = f"{problem_text} {' '.join(profile.columns)}".lower()
        if ("ahp" in text or "层次分析" in text) and "ahp_weights" not in candidates:
            self._add_candidate(candidates, "ahp_weights", "evaluation", 30, "题目明确包含 AHP 或层次分析", "AHP")
        if (profile.demand_columns or "demand" in text or "需求" in text) and (
            profile.capacity_columns or "capacity" in text or "容量" in text or "供给" in text
        ):
            self._add_candidate(candidates, "capacity_gap", "evaluation", 32, "数据或题目同时包含需求与容量", "供需缺口")

        if profile.binary_label_columns or profile.multiclass_label_columns:
            self._add_candidate(candidates, "logistic_classifier", "classification", 26, "data profile detected low-cardinality label columns", "classification")
            self._add_candidate(candidates, "knn_classifier", "classification", 18, "data profile detected low-cardinality label columns", "classification")
        if profile.has_edge_table:
            self._add_candidate(candidates, "graph_shortest_paths", "network", 28, "data profile detected source-target edge table", "edge table")
            self._add_candidate(candidates, "graph_centrality", "network", 24, "data profile detected source-target edge table", "edge table")
        if profile.demand_capacity_pairs:
            self._add_candidate(candidates, "capacity_gap", "evaluation", 36, "data profile detected paired demand-capacity fields", "capacity gap")
        if profile.has_objective_constraint_combo:
            self._add_candidate(candidates, "resource_allocation", "optimization", 30, "data profile detected objective and resource constraint fields", "resource optimization")
        if self._has_cement_esp_schema(profile):
            self._add_candidate(
                candidates,
                "cement_esp_optimization",
                "optimization",
                96,
                "检测到水泥窑 ESP 标准字段：入口/出口浓度、烟气流量、四电场电压与停留时间",
                "cement ESP optimization",
            )
            candidates["cement_esp_optimization"].semantic_score = max(
                candidates["cement_esp_optimization"].semantic_score,
                96,
            )

        explicit_hints = (
            (("esp", "电除尘", "除尘器", "烟尘", "水泥窑"), "cement_esp_optimization", "optimization"),
            (("capacity gap", "shortage", "供需缺口", "容量缺口"), "capacity_gap", "evaluation"),
            (("shortest path", "最短路"), "graph_shortest_paths", "network"),
            (("maximum flow", "max flow", "最大流"), "graph_max_flow", "network"),
            (("community", "communities", "社区", "社群"), "community_detection", "network"),
              (("hypothesis", "假设检验"), "hypothesis_tests", "statistics"),
              (("抽样检测", "抽检", "次品率", "不合格率", "接收", "拒收", "acceptance sampling"), "quality_sampling_plan", "statistics"),
            (("monte carlo", "蒙特卡洛"), "monte_carlo", "simulation"),
            (("assignment", "指派", "匹配"), "assignment_plan", "optimization"),
            (("scheduling", "schedule", "调度", "排程"), "scheduling_plan", "optimization"),
            (("knapsack", "背包"), "knapsack_01", "optimization"),
            (("crop", "planting", "farmland", "acreage"), "crop_planting_plan", "optimization"),
            (("seasonal", "季节"), "seasonal_forecast", "forecast"),
            (("pca", "主成分"), "pca", "clustering"),
            (("nipt", "bmi", "y chromosome", "gestational week"), "nipt_bmi_grouping", "statistics"),
        )
        for terms, model_id, task_type in explicit_hints:
            if any(term in text for term in terms):
                self._add_candidate(
                    candidates,
                    model_id,
                    task_type,
                    80
                    if model_id in {"graph_max_flow", "community_detection", "pca"}
                    else 44,
                    f"题目明确指定或强烈指向 {model_id}",
                    "explicit intent",
                )
                if model_id in {"graph_max_flow", "community_detection", "pca", "cement_esp_optimization"}:
                    candidates[model_id].semantic_score = max(
                        candidates[model_id].semantic_score, 90
                    )

        domain_hints = (
            (("标定", "参数辨识", "反演", "外延层厚度"), "parameter_identification", "statistics"),
            (("校准", "测量误差", "加权最小二乘"), "weighted_least_squares", "statistics"),
            (("高温作业", "炉温曲线", "热传导", "传热"), "heat_conduction", "simulation"),
            (("压力控制", "最优控制"), "optimal_control", "optimization"),
            (("波浪能", "振子", "阻尼振动"), "harmonic_oscillator", "simulation"),
            (("动态调度", "生产安排", "排班"), "scheduling_plan", "optimization"),
            (("补货", "库存策略"), "inventory_policy", "optimization"),
            (("季节性", "周期性", "未来一周"), "seasonal_forecast", "forecast"),
            (("交通流量", "车流量", "交通管控"), "traffic_flow", "network"),
            (("抽样检测", "信度", "次品率"), "quality_sampling_plan", "statistics"),
            (("干涉条纹", "红外光谱", "频谱"), "fft_frequency_analysis", "statistics"),
            (("定日镜", "测线", "重叠率"), "nonlinear_optimization", "optimization"),
            (("农作物", "种植策略", "地块", "亩产"), "crop_planting_plan", "optimization"),
            (("NIPT", "BMI", "胎儿异常"), "nipt_bmi_grouping", "statistics"),
            (("人体姿态", "关键节点", "跳远成绩"), "gradient_boosting", "forecast"),
            (("矿井突水", "水流漫延"), "bernoulli_flow", "simulation"),
            (("CT系统", "成像"), "image_registration", "classification"),
            (("中药材", "药材的鉴别"), "knn_classifier", "classification"),
            (("FAST", "主动反射面"), "nonlinear_optimization", "optimization"),
          )
        for terms, model_id, task_type in domain_hints:
            if any(term.lower() in text for term in terms):
                self._add_candidate(
                    candidates,
                    model_id,
                    task_type,
                    88,
                    f"国赛领域意图强匹配：{model_id}",
                    "competition domain intent",
                )
                candidates[model_id].semantic_score = max(
                    candidates[model_id].semantic_score,
                    88,
                )

    def _has_cement_esp_schema(self, profile: DataProfile) -> bool:
        columns = {column.lower() for column in profile.columns}
        required = {
            "temp_c",
            "c_in_gnm3",
            "q_nm3h",
            "u1_kv",
            "u2_kv",
            "u3_kv",
            "u4_kv",
            "t1_s",
            "t2_s",
            "t3_s",
            "t4_s",
            "c_out_mgnm3",
            "p_total_kw",
        }
        return required.issubset(columns)

    def _infer_task_type(self, entry: AlgorithmEntry) -> str:
        text = f"{entry.category} {entry.problem}".lower()
        if any(term in text for term in ("预测", "时间序列", "forecast")):
            return "forecast"
        if any(term in text for term in ("评价", "权重", "排序", "风险")):
            return "evaluation"
        if any(term in text for term in ("优化", "规划", "调度", "路径", "背包", "分配")):
            return "optimization"
        if any(term in text for term in ("分类", "聚类", "降维")):
            return "classification" if "分类" in text else "clustering"
        if any(term in text for term in ("图", "网络", "路径", "社区")):
            return "network"
        return "exploration"


class ModelSuitabilityAgent:
    """Score candidates against the data profile and expected task."""

    COMPARISON_HINTS: dict[str, tuple[str, ...]] = {
        "forecast": ("trend_forecast", "smoothing_forecast", "ridge_regression", "gradient_boosting"),
        "evaluation": ("entropy_weights", "topsis_rank", "grey_relation", "vikor"),
        "optimization": ("resource_allocation", "cement_esp_optimization", "knapsack_01", "integer_programming", "nonlinear_optimization"),
        "classification": ("logistic_classifier", "naive_bayes_classifier", "knn_classifier"),
        "clustering": ("kmeans_cluster", "dbscan_cluster", "hierarchical_cluster"),
        "network": ("graph_centrality", "graph_shortest_paths", "community_detection"),
        "statistics": ("correlation_analysis", "linear_regression", "hypothesis_tests", "quality_sampling_plan"),
        "simulation": ("monte_carlo", "sir_model", "logistic_growth"),
    }

    def run(self, candidates: tuple[CandidateModel, ...], profile: DataProfile) -> tuple[CandidateModel, ...]:
        for candidate in candidates:
            self._score(candidate, profile)
        return tuple(candidates)

    def _score(self, candidate: CandidateModel, profile: DataProfile) -> None:
        candidate.task_score += 12 if candidate.task_type != "exploration" else 4

        if not profile.has_data:
            candidate.risk_penalty += 18
            candidate.risks.append("未提供结构化数据，只能作为建模方案建议")
            return

        applicability = check_applicable(candidate.model_id, profile, candidate.task_type)
        contract = get_model_contract(candidate.model_id)
        candidate.contract = contract.as_dict()
        candidate.applicability = applicability.as_dict()
        candidate.role = applicability.role
        candidate.reasons.extend(applicability.reasons)
        candidate.risks.extend(applicability.warnings)
        if applicability.can_run:
            candidate.data_score += 14
        else:
            candidate.risk_penalty += 20 + 8 * len(applicability.required_fields)
        if applicability.required_fields:
            candidate.risks.append(f"required_fields={', '.join(applicability.required_fields)}")
        if applicability.task_types and candidate.task_type in applicability.task_types:
            candidate.task_score += 8
        elif candidate.task_type != "exploration":
            candidate.risk_penalty += 6

        numeric_count = len(profile.numeric_columns)
        if numeric_count:
            candidate.data_score += min(12, numeric_count * 3)
        else:
            candidate.risk_penalty += 14
            candidate.risks.append("未识别到数值字段")

        # Check not_recommended_when conditions from the applicability rule
        for condition in applicability.not_recommended_when:
            if self._condition_matches(condition, candidate.model_id, profile, candidate.task_type):
                candidate.risk_penalty += 10
                candidate.risks.append(f"禁忌条件命中: {condition}")

        # Apply tier-based scoring boosts
        tier = applicability.tier
        if tier == "innovation":
            candidate.tier = "innovation"
            candidate.innovation_score = min(10, candidate.innovation_score + 8)
            candidate.interpretability_score = min(10, candidate.interpretability_score + 1)
            candidate.reasons.append("tier=innovation — 模型为前沿/组合增强方案")
        elif tier == "improved":
            candidate.tier = "improved"
            candidate.innovation_score = min(10, candidate.innovation_score + 4)
            candidate.interpretability_score = min(10, candidate.interpretability_score + 2)
            candidate.reasons.append("tier=improved — 模型为标准改进方案")
        else:
            candidate.tier = "baseline"
            candidate.innovation_score = min(10, candidate.innovation_score + 1)
            candidate.interpretability_score = min(10, candidate.interpretability_score + 3)
            candidate.reasons.append("tier=baseline — 经典可解释基础模型")

        # Compute explainability_score based on model characteristics
        candidate.explainability_score = self._compute_explainability(candidate.model_id)
        # Compute applicability_score based on how well data fits the model
        candidate.applicability_score = self._compute_applicability_score(applicability, profile)
        # Augment innovation_score with innovation_extensions
        if applicability.innovation_extensions:
            candidate.innovation_score = min(10, candidate.innovation_score + len(applicability.innovation_extensions))
            candidate.reasons.append(f"创新扩展可用: {', '.join(applicability.innovation_extensions[:3])}")
        # Set validation plan from rule
        candidate.validation_plan = list(contract.diagnostics)
        # Store comparison candidates
        if candidate.task_type in self.COMPARISON_HINTS:
            candidate.comparison_candidates = [
                m for m in self.COMPARISON_HINTS.get(candidate.task_type, ())
                if m != candidate.model_id
            ][:3]
        if contract.baseline_models:
            candidate.comparison_candidates = list(
                dict.fromkeys(
                    [*contract.baseline_models, *candidate.comparison_candidates]
                )
            )[:4]

        self._apply_profile_signal_scores(candidate, profile)
        return

    def _apply_profile_signal_scores(self, candidate: CandidateModel, profile: DataProfile) -> None:
        if candidate.model_id in {"trend_forecast", "smoothing_forecast", "grey_gm11", "seasonal_forecast", "var_forecast"}:
            if profile.monotonic_time_columns:
                candidate.data_score += 12
                candidate.reasons.append(f"monotonic time columns: {', '.join(profile.monotonic_time_columns[:3])}")
            elif profile.datetime_columns:
                candidate.data_score += 8

        if candidate.model_id == "capacity_gap" and profile.demand_capacity_pairs:
            candidate.data_score += 12
            candidate.reasons.append("paired demand-capacity fields detected")

        if candidate.model_id in {"resource_allocation", "cement_esp_optimization", "knapsack_01", "scheduling_plan", "nonlinear_optimization", "integer_programming", "multiobjective_optimization"}:
            if profile.has_objective_constraint_combo:
                candidate.data_score += 14
                candidate.reasons.append("objective-resource constraint combination detected")
            if candidate.model_id == "cement_esp_optimization" and self._has_cement_esp_schema(profile):
                candidate.data_score += 28
                candidate.reasons.append("complete cement ESP process schema detected")

        if candidate.model_id in {"logistic_classifier", "naive_bayes_classifier", "knn_classifier", "smote_balance"}:
            if profile.binary_label_columns:
                candidate.data_score += 10
                candidate.reasons.append(f"binary label columns: {', '.join(profile.binary_label_columns[:3])}")
            elif profile.multiclass_label_columns:
                candidate.data_score += 8
                candidate.reasons.append(f"multiclass label columns: {', '.join(profile.multiclass_label_columns[:3])}")

        if candidate.model_id.startswith("graph_") or candidate.model_id in {"community_detection", "astar_path", "tsp_route", "vrp_route"}:
            if profile.has_edge_table:
                candidate.data_score += 12
                candidate.reasons.append("source-target edge table detected")

    def _has_cement_esp_schema(self, profile: DataProfile) -> bool:
        columns = {column.lower() for column in profile.columns}
        required = {
            "temp_c",
            "c_in_gnm3",
            "q_nm3h",
            "u1_kv",
            "u2_kv",
            "u3_kv",
            "u4_kv",
            "t1_s",
            "t2_s",
            "t3_s",
            "t4_s",
            "c_out_mgnm3",
            "p_total_kw",
        }
        return required.issubset(columns)

    def _condition_matches(self, condition: str, model_id: str, profile: DataProfile, task_type: str) -> bool:
        """Check if a not_recommended_when condition matches the current data/task context."""
        condition_lower = condition.lower()
        # Sample size based conditions
        if "large sample" in condition_lower or "大样本" in condition_lower:
            return profile.rows > 100
        if "small sample" in condition_lower or "小样本" in condition_lower or "fewer than" in condition_lower:
            return True  # flagged for review; small sample risk is inherent
        # Nonlinearity conditions
        if "nonlinear" in condition_lower and "nonlinear" not in condition_lower.replace("strong nonlinearity", ""):
            if "strong nonlinearity" in condition_lower:
                return profile.nonlinearity_score > 0.6
            return profile.nonlinearity_score > 0.4
        # Seasonality
        if "seasonality" in condition_lower or "seasonal" in condition_lower:
            return bool(profile.monotonic_time_columns) and profile.rows >= 12
        # High-dimensional
        if "high-dimensional" in condition_lower:
            return len(profile.numeric_columns) > 20
        # Interpretability
        if "interpretability" in condition_lower:
            return True  # always flag for human review if interpretability matters
        # Expert judgment
        if "expert" in condition_lower or "subjective" in condition_lower:
            return not profile.categorical_columns  # no subjective source
        # Correlated features
        if "correlated" in condition_lower or "collinearity" in condition_lower:
            return len(profile.numeric_columns) > 5  # plausible collinearity risk
        # Many explanatory variables
        if "many explanatory" in condition_lower:
            return len(profile.numeric_columns) > 8
        # Long forecast horizon
        if "long forecast" in condition_lower:
            return True  # always flag
        # Outliers
        if "outlier" in condition_lower:
            return profile.rows > 10  # flag when enough data to have outliers
        # Class imbalance
        if "balanced" in condition_lower:
            return bool(profile.binary_label_columns)
        # Data is crisp/clear
        if "crisp" in condition_lower or "clear boundar" in condition_lower:
            return profile.missing_rate < 0.02
        # Need absolute ranking
        if "absolute ranking" in condition_lower:
            return True
        # Overlapping communities
        if "overlapping" in condition_lower:
            return True  # flag for review
        # Dynamic
        if "dynamic" in condition_lower:
            return bool(profile.monotonic_time_columns)
        # Too many indicators
        if "too many indicators" in condition_lower:
            return len(profile.numeric_columns) > 15
        # Default: flag for human attention
        return True

    def _compute_explainability(self, model_id: str) -> int:
        """Compute 0-10 explainability score based on model complexity."""
        HIGH_EXPLAIN = {
            "trend_forecast", "smoothing_forecast", "linear_regression", "ridge_regression",
            "entropy_weights", "topsis_rank", "ahp_weights", "grey_relation",
            "correlation_analysis", "hypothesis_tests", "parameter_estimation", "nipt_bmi_grouping",
            "resource_allocation", "knapsack_01", "assignment_plan",
            "kmeans_cluster", "hierarchical_cluster", "graph_shortest_paths",
            "graph_mst", "graph_centrality", "capacity_gap",
        }
        MEDIUM_EXPLAIN = {
            "grey_gm11", "polynomial_fit", "logistic_classifier", "naive_bayes_classifier",
            "knn_classifier", "dbscan_cluster", "pca", "feature_selection",
            "anova_analysis", "monte_carlo", "scheduling_plan",
            "integer_programming", "nonlinear_optimization", "seasonal_forecast",
            "var_forecast", "vikor", "dea_efficiency", "fuzzy_evaluation",
            "graph_max_flow", "community_detection", "astar_path",
            "queue_metrics", "inventory_policy",
            "signal_denoising", "fft_frequency_analysis",
            "logistic_growth", "sir_model", "lotka_volterra", "solow_growth",
            "harmonic_oscillator", "michaelis_menten", "bernoulli_flow",
            "nash_equilibrium", "shapley_value",
            "kalman_filter", "optimal_control", "robust_control",
            "var_cvar_risk", "black_scholes_pricing",
            "markowitz_portfolio", "apriori_rules", "granger_causality",
        }
        LOW_EXPLAIN = {
            "gradient_boosting", "nonlinear_forecast", "multiobjective_optimization",
            "nonlinear_embedding", "smote_balance", "ahp_entropy_combined",
            "tsp_route", "vrp_route", "bin_packing",
            "garch_volatility", "auction_pricing", "stackelberg_equilibrium",
            "bullwhip_effect", "multi_echelon_inventory", "jackson_network",
            "image_features", "image_registration", "image_segmentation",
            "traffic_flow", "car_following", "energy_detection",
            "heat_conduction", "weighted_least_squares", "nonlinear_fit",
            "parameter_identification", "edge_detection", "histogram_equalization",
        }
        if model_id in HIGH_EXPLAIN:
            return 9
        if model_id in MEDIUM_EXPLAIN:
            return 6
        if model_id in LOW_EXPLAIN:
            return 3
        return 5

    def _compute_applicability_score(self, applicability: Any, profile: DataProfile) -> int:
        """Compute 0-10 score for how well the model fits the data conditions."""
        score = 5  # neutral start
        if applicability.can_run:
            score += 3
        else:
            score -= 5
        num_warnings = len(applicability.warnings)
        score -= min(4, num_warnings)
        num_reasons = len(applicability.reasons)
        score += min(3, num_reasons)
        if profile.nonlinearity_score < 0.3:
            score += 1  # near-linear data is well-behaved
        if profile.missing_rate < 0.05:
            score += 1  # clean data
        return max(0, min(10, score))

class SelectionSynthesisAgent:
    """Choose final models and build machine-readable plus Markdown reports."""

    MAX_MODELS = 14

    def run(
        self,
        tasks: tuple[SelectionTask, ...],
        profile: DataProfile,
        candidates: tuple[CandidateModel, ...],
        references: tuple[ReferenceChoice, ...],
    ) -> ModelSelectionResult:
        ordered = sorted(candidates, key=lambda item: (item.total_score, item.semantic_score), reverse=True)
        selected_list: list[CandidateModel] = []
        explicit_candidates = [
            item
            for item in ordered
            if item.semantic_score >= 60 and self._can_run(item, profile)
        ]
        explicit_primary = max(
            explicit_candidates,
            key=lambda item: (item.semantic_score, item.total_score),
            default=None,
        )
        if explicit_primary is not None:
            self._append_unique(selected_list, explicit_primary)
        for task in tasks:
            task_candidates = [
                item
                for item in ordered
                if item.total_score >= 12
                and self._can_run(item, profile)
                and (item.task_type == task.task_type or task.task_id in item.task_ids)
            ]
            if task_candidates:
                primary = task_candidates[0]
                self._append_unique(selected_list, primary)
                contract = get_model_contract(primary.model_id)
                for baseline_id in contract.baseline_models:
                    baseline = next(
                        (
                            item
                            for item in ordered
                            if item.model_id == baseline_id
                            and self._can_run(item, profile)
                        ),
                        None,
                    )
                    if baseline is not None:
                        self._append_unique(selected_list, baseline)
                        break

        for preferred_role in ("primary", "comparison", "validation"):
            for item in ordered:
                if len(selected_list) >= self.MAX_MODELS:
                    break
                if (
                    item.total_score >= 12
                    and self._can_run(item, profile)
                    and item.role == preferred_role
                ):
                    self._append_unique(selected_list, item)

        selected = tuple(selected_list[: self.MAX_MODELS])
        if not selected and profile.has_data:
            selected = tuple(self._fallback_candidates(profile))
        rejected = tuple(item for item in ordered if item not in selected)
        report = self._format_report(tasks, profile, selected, rejected, references)
        report_json = self._format_json(tasks, profile, selected, rejected, references)
        return ModelSelectionResult(tasks, profile, selected, rejected, references, report, report_json)

    def _append_unique(self, selected: list[CandidateModel], item: CandidateModel) -> None:
        if all(existing.model_id != item.model_id for existing in selected):
            selected.append(item)

    def _can_run(self, item: CandidateModel, profile: DataProfile) -> bool:
        if item.applicability.get("can_run", True):
            return True
        if item.semantic_score >= 85 and any(
            "国赛领域意图强匹配" in reason for reason in item.reasons
        ):
            return True
        return not profile.has_data and item.semantic_score >= 60

    def _fallback_candidates(self, profile: DataProfile) -> list[CandidateModel]:
        fallback: list[CandidateModel] = []
        for model_id in ("trend_forecast", "entropy_weights", "topsis_rank"):
            applicability = check_applicable(model_id, profile, "exploration")
            if not applicability.can_run:
                continue
            fallback.append(
                CandidateModel(
                    model_id=model_id,
                    label=EXECUTABLE_MODEL_LABELS[model_id],
                    task_type="exploration",
                    semantic_score=8,
                    data_score=8 if profile.numeric_columns else 0,
                    task_score=4,
                    role=applicability.role,
                    reasons=["未命中明确题型，使用基础探索模型兜底"],
                    source_problem="默认探索",
                )
            )
        return fallback

    def _format_report(
        self,
        tasks: tuple[SelectionTask, ...],
        profile: DataProfile,
        selected: tuple[CandidateModel, ...],
        rejected: tuple[CandidateModel, ...],
        references: tuple[ReferenceChoice, ...],
    ) -> str:
        lines = [
            "# 模型选择",
            "",
            f"- 已接入算法目录条目数：{catalog_size()}",
            f"- 当前可直接运行模型数：{len(EXECUTABLE_MODEL_LABELS)}",
            f"- 数据画像：{profile.rows} 行，{len(profile.columns)} 列，数值字段 {len(profile.numeric_columns)} 个，非线性度 {profile.nonlinearity_score}，缺失率 {profile.missing_rate}，样本量等级 {profile.sample_size_category}",
            f"- 识别字段：时间 {len(profile.datetime_columns)} 个，目标 {len(profile.target_columns)} 个，约束 {len(profile.constraint_columns)} 个，关系 {len(profile.relation_columns)} 个",
            "",
            "## 五智能体协作结果",
            "- 任务拆解 Agent：识别题目需要回答的建模子任务",
            "- 数据画像 Agent：识别字段类型、时间列、需求/容量/成本/收益和关系字段",
            "- 候选生成 Agent：结合算法目录和任务类型生成候选模型",
            "- 适用性评分 Agent：按数据条件、风险和可解释性打分",
            "- 选择汇总 Agent：输出主模型、对比模型、验证模型和拒绝原因",
            "",
            "## 任务拆解",
        ]
        for task in tasks:
            evidence = "、".join(task.evidence) if task.evidence else "默认探索"
            lines.append(f"- {task.task_id}：{task.task_type}，{task.goal}；证据：{evidence}")

        lines.extend(["", "## 可执行模型"])
        if selected:
            for item in selected:
                reason = "；".join(item.reasons[:2])
                risks = f"；风险：{'；'.join(item.risks[:2])}" if item.risks else ""
                task_ids = f"，任务={','.join(item.task_ids)}" if item.task_ids else ""
                lines.append(f"- {item.label}（`{item.model_id}`）：{item.role}{task_ids}，score={item.total_score}，tier={item.tier}，{reason}{risks}")
        else:
            lines.append("- 未选择可执行模型。")

        lines.extend(["", "## 未采用或降级的候选"])
        if rejected:
            for item in rejected[:8]:
                risks = "；".join(item.risks) if item.risks else "分数低于入选模型"
                lines.append(f"- {item.label}（`{item.model_id}`）：score={item.total_score}，{risks}")
        else:
            lines.append("- 无。")

        lines.extend(["", "## 参考算法推荐"])
        if references:
            for item in references:
                recommended = "、".join(item.recommended)
                alternatives = "、".join(item.alternatives) if item.alternatives else "无"
                lines.append(f"- {item.category} / {item.problem}：推荐 {recommended}；备选 {alternatives}")
        else:
            lines.append("- 未命中算法目录中的具体题型，可补充题目关键词或数据说明。")

        lines.extend(
            [
                "",
                "## 说明",
                "- 选模不再只看关键词；关键词命中只贡献语义分，最终会被数据适用性和风险修正。",
                "- 低分模型不会进入执行链路，减少无关模型堆叠和空结果表。",
            ]
        )
        return "\n".join(lines)

    def _format_json(
        self,
        tasks: tuple[SelectionTask, ...],
        profile: DataProfile,
        selected: tuple[CandidateModel, ...],
        rejected: tuple[CandidateModel, ...],
        references: tuple[ReferenceChoice, ...],
    ) -> str:
        subproblem_models = self._subproblem_models(tasks, selected)
        payload: dict[str, Any] = {
            "tasks": [asdict(task) for task in tasks],
            "data_profile": asdict(profile),
            "selected_model_ids": [item.model_id for item in selected],
            "subproblem_models": subproblem_models,
            "selected_models": [self._candidate_dict(item) for item in selected],
            "rejected_models": [self._candidate_dict(item) for item in rejected],
            "reference_algorithms": [asdict(item) for item in references],
            "model_comparison_plan": self._comparison_plan(selected),
            "innovation_recommendations": self._innovation_summary(selected),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _subproblem_models(
        self,
        tasks: tuple[SelectionTask, ...],
        selected: tuple[CandidateModel, ...],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for task in tasks:
            matches = [
                item
                for item in selected
                if task.task_id in item.task_ids or (task.task_id not in item.task_ids and item.task_type == task.task_type)
            ]
            if not matches:
                matches = [item for item in selected if item.task_type == task.task_type]

            buckets = {
                "primary": [self._candidate_dict(item) for item in matches if item.role == "primary"],
                "comparison": [self._candidate_dict(item) for item in matches if item.role == "comparison"],
                "validation": [self._candidate_dict(item) for item in matches if item.role == "validation"],
            }
            grouped[task.task_id] = {
                "task": asdict(task),
                **buckets,
            }
        return grouped

    def _candidate_dict(self, item: CandidateModel) -> dict[str, Any]:
        payload = asdict(item)
        payload["total_score"] = item.total_score
        return payload

    def _comparison_plan(self, selected: tuple[CandidateModel, ...]) -> list[dict[str, Any]]:
        """Generate a model comparison plan for each selected model."""
        plans: list[dict[str, Any]] = []
        for item in selected:
            plan = {
                "model_id": item.model_id,
                "label": item.label,
                "task_type": item.task_type,
                "comparison_candidates": item.comparison_candidates,
                "validation_plan": item.validation_plan,
                "metrics": self._metrics_for_task_type(item.task_type),
            }
            plans.append(plan)
        return plans

    def _metrics_for_task_type(self, task_type: str) -> list[str]:
        """Return appropriate validation metrics for a task type."""
        metric_map = {
            "forecast": ["MAE", "RMSE", "MAPE", "R²", "rolling_cv_error"],
            "evaluation": ["ranking_stability", "weight_sensitivity", "spearman_correlation"],
            "optimization": ["objective_value", "constraint_violation", "resource_utilization", "robustness"],
            "classification": ["accuracy", "f1_score", "auc", "confusion_matrix"],
            "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin"],
            "network": ["modularity", "connectivity", "diameter", "avg_path_length"],
            "statistics": ["p_value", "effect_size", "confidence_interval", "r_squared"],
            "simulation": ["convergence", "sensitivity", "monte_carlo_error"],
        }
        return metric_map.get(task_type, ["rmse", "mae", "r_squared"])

    def _innovation_summary(self, selected: tuple[CandidateModel, ...]) -> list[dict[str, Any]]:
        """Summarize innovation recommendations from selected models."""
        innovations: list[dict[str, Any]] = []
        for item in selected:
            if item.tier == "innovation" or item.innovation_score >= 6:
                innovations.append({
                    "model_id": item.model_id,
                    "label": item.label,
                    "task_type": item.task_type,
                    "tier": item.tier,
                    "innovation_score": item.innovation_score,
                    "innovation_extensions": item.applicability.get("innovation_extensions", []),
                    "reason": f"采用 {item.tier} 级方案，创新评分 {item.innovation_score}/10",
                })
        return innovations


class ModelSelectionCrew:
    def __init__(self, llm: Any | None = None) -> None:
        self.task_agent = TaskDecompositionAgent(llm=llm)
        self.profile_agent = DataProfileAgent()
        self.candidate_agent = CandidateModelAgent()
        self.suitability_agent = ModelSuitabilityAgent()
        self.synthesis_agent = SelectionSynthesisAgent()
        self.innovation_agent = InnovationRecommendationAgent()

    def run(self, problem_text: str, data_files: list[Path], columns: list[str]) -> ModelSelectionResult:
        tasks = self.task_agent.run(problem_text, columns)
        profile = self.profile_agent.run(data_files)
        if not profile.columns and columns:
            profile = DataProfile(
                columns=tuple(columns),
                rows=0,
                numeric_columns=(),
                categorical_columns=(),
                datetime_columns=(),
                id_like_columns=(),
                demand_columns=(),
                capacity_columns=(),
                cost_columns=(),
                benefit_columns=(),
                relation_columns=(),
                target_columns=(),
                constraint_columns=(),
                source_files=tuple(str(path) for path in data_files),
            )
        candidates = self.candidate_agent.run(problem_text, profile, tasks)
        scored = self.suitability_agent.run(candidates, profile)
        references = self.candidate_agent.references(problem_text, list(profile.columns))
        result = self.synthesis_agent.run(tasks, profile, scored, references)
        # Build result with innovation suggestions
        innovation_suggestions = self.innovation_agent.run(tasks, profile, scored)
        return ModelSelectionResult(
            tasks=result.tasks,
            data_profile=result.data_profile,
            selected=result.selected,
            rejected=result.rejected,
            references=result.references,
            report=result.report,
            report_json=result.report_json,
            innovation_suggestions=innovation_suggestions,
        )
