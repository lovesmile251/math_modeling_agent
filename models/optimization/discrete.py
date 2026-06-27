from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def knapsack_01_plan(df: pd.DataFrame, capacity_ratio: float = 0.5) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 1 or numeric.shape[1] < 2:
        return pd.DataFrame()

    weight_col = _find_column(numeric, ("weight", "cost", "resource", "volume", "time", "重量", "成本", "资源", "体积", "时间"))
    value_col = _find_column(numeric, ("value", "profit", "benefit", "score", "revenue", "价值", "收益", "利润", "得分"), exclude={weight_col})
    capacity_col = _find_column(numeric, ("capacity", "budget", "limit", "容量", "预算", "上限"), exclude={weight_col, value_col})
    if weight_col is None:
        weight_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [column for column in numeric.columns if str(column) != weight_col]
        if not candidates:
            return pd.DataFrame()
        value_col = str(candidates[-1])

    work = pd.DataFrame(
        {
            "row_index": numeric.index,
            "weight": pd.to_numeric(numeric[weight_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
        }
    ).dropna()
    work = work[(work["weight"] > 0) & (work["value"] > 0)].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        capacity = float(work["weight"].sum() * capacity_ratio)
    if capacity <= 0:
        return pd.DataFrame()

    selected_positions = _solve_knapsack_positions(work["weight"].to_numpy(float), work["value"].to_numpy(float), capacity)
    selected = work.iloc[selected_positions].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["selected"] = 1
    selected["capacity"] = capacity
    selected["total_weight"] = float(selected["weight"].sum())
    selected["total_value"] = float(selected["value"].sum())
    selected["weight_column"] = str(weight_col)
    selected["value_column"] = str(value_col)
    selected["method"] = "0-1 dynamic programming" if len(work) <= 200 else "value density greedy"
    return selected[
        [
            "row_index",
            "selected",
            "weight_column",
            "value_column",
            "weight",
            "value",
            "capacity",
            "total_weight",
            "total_value",
            "method",
        ]
    ]


def assignment_plan(df: pd.DataFrame) -> pd.DataFrame:
    long_form = _assignment_long_form(df)
    if long_form.empty:
        return pd.DataFrame()

    agents = list(dict.fromkeys(long_form["agent"].astype(str)))
    tasks = list(dict.fromkeys(long_form["task"].astype(str)))
    if not agents or not tasks:
        return pd.DataFrame()

    cost_map = {
        (str(row.agent), str(row.task)): float(row.cost)
        for row in long_form.itertuples(index=False)
        if pd.notna(row.cost)
    }
    assignments = _solve_assignment(agents, tasks, cost_map)
    rows = []
    total_cost = sum(item[2] for item in assignments)
    for rank, (agent, task, cost) in enumerate(assignments, start=1):
        rows.append(
            {
                "assignment_order": rank,
                "agent": agent,
                "task": task,
                "cost": float(cost),
                "total_cost": float(total_cost),
                "method": "bruteforce_min_cost" if len(agents) <= 8 and len(tasks) <= 8 else "greedy_min_cost",
            }
        )
    return pd.DataFrame(rows)


def bin_packing_plan(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()

    size_col = _find_column(numeric, ("size", "weight", "volume", "load", "重量", "体积", "尺寸", "载重"))
    capacity_col = _find_column(numeric, ("capacity", "bin_capacity", "limit", "容量", "箱容", "上限"), exclude={size_col})
    if size_col is None:
        size_col = str(numeric.columns[0])

    sizes = pd.to_numeric(numeric[size_col], errors="coerce").dropna()
    sizes = sizes[sizes > 0]
    if sizes.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        capacity = float(max(sizes.max(), sizes.mean() * 2.5))
    if capacity <= 0:
        return pd.DataFrame()

    items = sorted([(int(idx), float(size)) for idx, size in sizes.items()], key=lambda item: item[1], reverse=True)
    bins: list[dict[str, float | list[int]]] = []
    rows: list[dict[str, float | str | int]] = []
    for row_index, size in items:
        best_bin = None
        best_remaining = math.inf
        for idx, bin_item in enumerate(bins):
            remaining = float(bin_item["remaining"])
            if size <= remaining and remaining - size < best_remaining:
                best_bin = idx
                best_remaining = remaining - size
        if best_bin is None:
            bins.append({"remaining": capacity - size, "load": size, "items": [row_index]})
            bin_id = len(bins)
        else:
            bins[best_bin]["remaining"] = float(bins[best_bin]["remaining"]) - size
            bins[best_bin]["load"] = float(bins[best_bin]["load"]) + size
            cast_items = bins[best_bin]["items"]
            if isinstance(cast_items, list):
                cast_items.append(row_index)
            bin_id = best_bin + 1
        rows.append(
            {
                "row_index": row_index,
                "item_size": size,
                "bin_id": bin_id,
                "bin_capacity": capacity,
                "method": "best_fit_decreasing",
            }
        )

    result = pd.DataFrame(rows).sort_values(["bin_id", "row_index"]).reset_index(drop=True)
    loads = result.groupby("bin_id")["item_size"].sum().rename("bin_load")
    result = result.join(loads, on="bin_id")
    result["bin_utilization"] = result["bin_load"] / result["bin_capacity"]
    result["bin_count"] = result["bin_id"].nunique()
    return result


def scheduling_plan(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()

    duration_col = _find_column(numeric, ("duration", "process", "time", "processing", "工期", "时长", "加工", "时间"))
    due_col = _find_column(numeric, ("due", "deadline", "交期", "截止", "期限"), exclude={duration_col})
    priority_col = _find_column(numeric, ("priority", "weight", "importance", "优先级", "权重", "重要"), exclude={duration_col, due_col})
    if duration_col is None:
        duration_col = str(numeric.columns[0])

    work = pd.DataFrame(
        {
            "row_index": numeric.index,
            "duration": pd.to_numeric(numeric[duration_col], errors="coerce"),
            "due": pd.to_numeric(numeric[due_col], errors="coerce") if due_col else np.nan,
            "priority": pd.to_numeric(numeric[priority_col], errors="coerce") if priority_col else 1.0,
        }
    ).dropna(subset=["duration"])
    work = work[work["duration"] > 0]
    if work.empty:
        return pd.DataFrame()

    if due_col:
        work["due"] = work["due"].fillna(float(work["due"].max()))
        work = work.sort_values(["due", "duration"], ascending=[True, True])
        method = "earliest_due_date"
    elif priority_col:
        work["priority"] = work["priority"].fillna(1.0)
        work = work.sort_values(["priority", "duration"], ascending=[False, True])
        method = "priority_dispatch"
    else:
        work = work.sort_values("duration", ascending=True)
        method = "shortest_processing_time"

    rows = []
    current_time = 0.0
    for order, row in enumerate(work.itertuples(index=False), start=1):
        start = current_time
        end = start + float(row.duration)
        due = float(row.due) if pd.notna(row.due) else float("nan")
        tardiness = max(0.0, end - due) if pd.notna(row.due) else float("nan")
        rows.append(
            {
                "sequence": order,
                "row_index": int(row.row_index),
                "start_time": start,
                "end_time": end,
                "duration": float(row.duration),
                "due_time": due,
                "tardiness": tardiness,
                "priority": float(row.priority) if pd.notna(row.priority) else 1.0,
                "method": method,
            }
        )
        current_time = end
    result = pd.DataFrame(rows)
    result["makespan"] = current_time
    result["total_tardiness"] = float(result["tardiness"].fillna(0.0).sum())
    return result


def _solve_knapsack_positions(weights: np.ndarray, values: np.ndarray, capacity: float) -> list[int]:
    if len(weights) > 200:
        order = np.argsort(-(values / weights))
        selected: list[int] = []
        used = 0.0
        for idx in order:
            if used + weights[idx] <= capacity:
                selected.append(int(idx))
                used += float(weights[idx])
        return selected

    scale = _integer_scale(weights, capacity)
    int_weights = np.maximum(1, np.round(weights * scale).astype(int))
    int_capacity = int(round(capacity * scale))
    if int_capacity <= 0 or int_capacity > 10000:
        order = np.argsort(-(values / weights))
        selected = []
        used = 0.0
        for idx in order:
            if used + weights[idx] <= capacity:
                selected.append(int(idx))
                used += float(weights[idx])
        return selected

    dp = np.zeros((len(weights) + 1, int_capacity + 1), dtype=float)
    keep = np.zeros((len(weights), int_capacity + 1), dtype=bool)
    for i, (weight, value) in enumerate(zip(int_weights, values), start=1):
        dp[i] = dp[i - 1]
        if weight <= int_capacity:
            candidate = dp[i - 1, : int_capacity + 1 - weight] + value
            better = candidate > dp[i, weight:]
            dp[i, weight:][better] = candidate[better]
            keep[i - 1, weight:][better] = True

    selected_positions: list[int] = []
    remaining = int_capacity
    for i in range(len(weights) - 1, -1, -1):
        if keep[i, remaining]:
            selected_positions.append(i)
            remaining -= int_weights[i]
    return list(reversed(selected_positions))


def _integer_scale(weights: np.ndarray, capacity: float) -> int:
    values = np.append(weights, capacity)
    decimals = 0
    for value in values:
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        if "." in text:
            decimals = max(decimals, len(text.split(".")[1]))
    return 10 ** min(decimals, 2)


def _assignment_long_form(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {str(column).lower(): column for column in df.columns}
    agent_col = next((lower_cols[name] for name in lower_cols if name in {"agent", "worker", "person", "machine", "人", "工人", "机器"}), None)
    task_col = next((lower_cols[name] for name in lower_cols if name in {"task", "job", "project", "任务", "项目", "工作"}), None)
    cost_col = next((lower_cols[name] for name in lower_cols if name in {"cost", "time", "distance", "score", "成本", "时间", "距离", "得分"}), None)
    if agent_col is not None and task_col is not None and cost_col is not None:
        return pd.DataFrame(
            {
                "agent": df[agent_col].astype(str),
                "task": df[task_col].astype(str),
                "cost": pd.to_numeric(df[cost_col], errors="coerce"),
            }
        ).dropna(subset=["cost"])

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 1 or numeric.shape[1] < 1:
        return pd.DataFrame()
    rows = []
    for row_idx, row in numeric.iterrows():
        for column in numeric.columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                rows.append({"agent": f"row_{row_idx}", "task": str(column), "cost": float(value)})
    return pd.DataFrame(rows)


def _solve_assignment(agents: list[str], tasks: list[str], cost_map: dict[tuple[str, str], float]) -> list[tuple[str, str, float]]:
    usable_count = min(len(agents), len(tasks))
    if usable_count <= 8:
        best: tuple[float, tuple[str, ...]] | None = None
        for task_perm in itertools.permutations(tasks, usable_count):
            cost = 0.0
            feasible = True
            for agent, task in zip(agents[:usable_count], task_perm):
                item_cost = cost_map.get((agent, task))
                if item_cost is None:
                    feasible = False
                    break
                cost += item_cost
            if feasible and (best is None or cost < best[0]):
                best = (cost, task_perm)
        if best is not None:
            return [(agent, task, cost_map[(agent, task)]) for agent, task in zip(agents[:usable_count], best[1])]

    available_tasks = set(tasks)
    assignments: list[tuple[str, str, float]] = []
    for agent in agents:
        candidates = [(task, cost_map[(agent, task)]) for task in available_tasks if (agent, task) in cost_map]
        if not candidates:
            continue
        task, cost = min(candidates, key=lambda item: item[1])
        assignments.append((agent, task, cost))
        available_tasks.remove(task)
        if not available_tasks:
            break
    return assignments


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    exclude = exclude or set()
    for column in df.columns:
        if str(column) in exclude:
            continue
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None
