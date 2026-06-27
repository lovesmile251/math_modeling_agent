from __future__ import annotations

import numpy as np
import pandas as pd


OBJECTIVE_KEYWORDS = (
    "profit",
    "benefit",
    "revenue",
    "value",
    "score",
    "output",
    "demand",
    "capacity",
    "收益",
    "利润",
    "价值",
    "得分",
    "产出",
    "需求",
    "容量",
)
RESOURCE_KEYWORDS = (
    "cost",
    "expense",
    "resource",
    "budget",
    "time",
    "distance",
    "risk",
    "成本",
    "费用",
    "资源",
    "预算",
    "时间",
    "距离",
    "风险",
)


def resource_allocation_plan(df: pd.DataFrame, budget_ratio: float = 0.6) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[0] == 0 or numeric.shape[1] < 2:
        return pd.DataFrame()

    resource_column = _find_column(numeric, RESOURCE_KEYWORDS)
    objective_column = _find_column(numeric, OBJECTIVE_KEYWORDS, exclude={resource_column} if resource_column else set())
    if resource_column is None:
        resource_column = _choose_resource_column(numeric)
    if objective_column is None:
        objective_column = _choose_objective_column(numeric, exclude={resource_column})
    if objective_column is None or resource_column is None or objective_column == resource_column:
        return pd.DataFrame()

    work = pd.DataFrame(
        {
            "row_index": numeric.index,
            "objective": pd.to_numeric(numeric[objective_column], errors="coerce"),
            "resource": pd.to_numeric(numeric[resource_column], errors="coerce"),
        }
    ).dropna()
    work = work[work["resource"] > 0]
    if work.empty:
        return pd.DataFrame()

    total_resource = float(work["resource"].sum())
    budget = max(total_resource * budget_ratio, float(work["resource"].min()))
    work["value_per_resource"] = work["objective"] / work["resource"]
    work = work.sort_values("value_per_resource", ascending=False).reset_index(drop=True)

    remaining = budget
    allocations: list[float] = []
    for resource in work["resource"]:
        allocation = float(min(resource, max(remaining, 0.0)))
        allocations.append(allocation)
        remaining -= allocation

    work["allocated_resource"] = allocations
    work["allocation_ratio"] = np.divide(
        work["allocated_resource"],
        work["resource"],
        out=np.zeros(len(work), dtype=float),
        where=work["resource"].to_numpy(dtype=float) != 0,
    )
    work["estimated_objective"] = work["objective"] * work["allocation_ratio"]
    work["priority_rank"] = np.arange(1, len(work) + 1)
    work["objective_column"] = str(objective_column)
    work["resource_column"] = str(resource_column)
    work["budget"] = budget
    work["method"] = "continuous_greedy_linear_allocation"
    return work[
        [
            "priority_rank",
            "row_index",
            "objective_column",
            "resource_column",
            "objective",
            "resource",
            "value_per_resource",
            "budget",
            "allocated_resource",
            "allocation_ratio",
            "estimated_objective",
            "method",
        ]
    ]


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    exclude = exclude or set()
    for column in df.columns:
        if str(column) in exclude:
            continue
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None


def _choose_resource_column(df: pd.DataFrame) -> str | None:
    positive_columns = [column for column in df.columns if (pd.to_numeric(df[column], errors="coerce") > 0).any()]
    if not positive_columns:
        return None
    # Lower variation is a conservative proxy for a resource consumption or cost column.
    return min(positive_columns, key=lambda column: float(pd.to_numeric(df[column], errors="coerce").std() or 0.0))


def _choose_objective_column(df: pd.DataFrame, exclude: set[str | None]) -> str | None:
    candidates = [column for column in df.columns if str(column) not in exclude]
    if not candidates:
        return None
    return max(candidates, key=lambda column: float(pd.to_numeric(df[column], errors="coerce").std() or 0.0))
