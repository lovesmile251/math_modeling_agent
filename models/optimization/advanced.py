from __future__ import annotations

from collections import defaultdict
import heapq
import itertools
import math

import numpy as np
import pandas as pd


BENEFIT_KEYWORDS = ("profit", "benefit", "revenue", "value", "score", "utility", "return", "output")
COST_KEYWORDS = ("cost", "loss", "error", "risk", "distance", "time", "expense", "penalty")


def nonlinear_gradient_optimization(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 4 or numeric.shape[1] < 2:
        return pd.DataFrame()

    objective_col = _find_column(numeric, BENEFIT_KEYWORDS + COST_KEYWORDS + ("objective", "target"))
    if objective_col is None:
        objective_col = str(numeric.columns[-1])
    variable_cols = [str(column) for column in numeric.columns if str(column) != objective_col]
    if not variable_cols:
        return pd.DataFrame()
    variable_cols = variable_cols[:6]

    work = numeric[[*variable_cols, objective_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if work.shape[0] < max(4, len(variable_cols) + 2):
        return pd.DataFrame()

    x_raw = work[variable_cols].to_numpy(dtype=float)
    y = work[objective_col].to_numpy(dtype=float)
    center = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x_raw - center) / scale
    design = _quadratic_design(x)
    try:
        coef = np.linalg.pinv(design) @ y
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    maximize = _looks_like(objective_col, BENEFIT_KEYWORDS) and not _looks_like(objective_col, COST_KEYWORDS)
    sign = -1.0 if maximize else 1.0
    z = np.zeros(len(variable_cols), dtype=float)
    learning_rate = 0.08
    previous = sign * _quadratic_predict(z, coef)
    for _ in range(250):
        grad = sign * _quadratic_gradient(z, coef)
        if not np.all(np.isfinite(grad)):
            return pd.DataFrame()
        candidate = np.clip(z - learning_rate * grad, -3.0, 3.0)
        value = sign * _quadratic_predict(candidate, coef)
        if value <= previous:
            if np.linalg.norm(candidate - z) < 1e-7:
                z = candidate
                break
            z = candidate
            previous = value
            learning_rate = min(learning_rate * 1.05, 0.5)
        else:
            learning_rate *= 0.5
            if learning_rate < 1e-6:
                break

    optimum = center + z * scale
    objective_value = float(_quadratic_predict(z, coef))
    gradient = _quadratic_gradient(z, coef) / scale
    rows = []
    for name, start, value, item_gradient in zip(variable_cols, center, optimum, gradient):
        rows.append(
            {
                "variable": name,
                "initial_value": float(start),
                "optimized_value": float(value),
                "gradient": float(item_gradient),
                "objective_column": str(objective_col),
                "predicted_objective": objective_value,
                "direction": "maximize" if maximize else "minimize",
                "method": "quadratic_surrogate_gradient_descent",
            }
        )
    return pd.DataFrame(rows)


def integer_branch_bound(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 1 or numeric.shape[1] < 2:
        return pd.DataFrame()

    weight_col = _find_column(numeric, ("weight", "cost", "resource", "volume", "time", "size"))
    value_col = _find_column(numeric, BENEFIT_KEYWORDS, exclude={weight_col})
    capacity_col = _find_column(numeric, ("capacity", "budget", "limit", "bound"), exclude={weight_col, value_col})
    if weight_col is None:
        weight_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != weight_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

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
        capacity = float(work["weight"].sum() * 0.5)
    if capacity <= 0:
        return pd.DataFrame()

    weights = work["weight"].to_numpy(dtype=float)
    values = work["value"].to_numpy(dtype=float)
    selected = _branch_bound_knapsack(weights, values, capacity) if len(work) <= 48 else _greedy_knapsack(weights, values, capacity)
    if not selected:
        return pd.DataFrame()

    result = work.iloc[selected].copy()
    result["selected"] = 1
    result["capacity"] = capacity
    result["total_weight"] = float(result["weight"].sum())
    result["total_value"] = float(result["value"].sum())
    result["weight_column"] = str(weight_col)
    result["value_column"] = str(value_col)
    result["method"] = "branch_and_bound_01_integer" if len(work) <= 48 else "density_greedy_integer"
    return result[
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
    ].reset_index(drop=True)


def multiobjective_weighted_sum(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 1 or numeric.shape[1] < 2:
        return pd.DataFrame()

    criteria = [str(column) for column in numeric.columns if not _looks_like(str(column), ("id", "index", "rank"))]
    criteria = [column for column in criteria if not column.lower().startswith("weight")]
    if len(criteria) < 2:
        return pd.DataFrame()

    work = numeric[criteria].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if work.empty:
        return pd.DataFrame()

    weights = _criteria_weights(df, criteria)
    normalized = pd.DataFrame(index=work.index)
    directions: dict[str, str] = {}
    for column in criteria:
        series = pd.to_numeric(work[column], errors="coerce")
        if series.dropna().empty:
            continue
        direction = "cost" if _looks_like(column, COST_KEYWORDS) else "benefit"
        directions[column] = direction
        span = float(series.max() - series.min())
        if span <= 1e-12:
            normalized[column] = 1.0
        elif direction == "cost":
            normalized[column] = (series.max() - series) / span
        else:
            normalized[column] = (series - series.min()) / span
    if normalized.empty:
        return pd.DataFrame()

    weight_vector = np.array([weights.get(column, 0.0) for column in normalized.columns], dtype=float)
    total = weight_vector.sum()
    if total <= 0:
        weight_vector = np.ones(len(normalized.columns), dtype=float) / len(normalized.columns)
    else:
        weight_vector = weight_vector / total

    scores = normalized.fillna(0.0).to_numpy(dtype=float) @ weight_vector
    result = pd.DataFrame({"row_index": normalized.index, "weighted_score": scores})
    result["rank"] = result["weighted_score"].rank(ascending=False, method="min").astype(int)
    result["criteria"] = ",".join(str(column) for column in normalized.columns)
    result["weights"] = ",".join(f"{column}:{weight:.4f}" for column, weight in zip(normalized.columns, weight_vector))
    result["directions"] = ",".join(f"{column}:{directions.get(column, 'benefit')}" for column in normalized.columns)
    result["method"] = "normalized_weighted_sum"
    return result.sort_values(["rank", "row_index"]).reset_index(drop=True)


def astar_path_plan(df: pd.DataFrame) -> pd.DataFrame:
    edges = _extract_edges(df)
    if not edges.empty:
        return _astar_edges(edges, df)
    grid = _extract_grid(df)
    if not grid.empty:
        return _astar_grid(grid)
    return pd.DataFrame()


def tsp_route_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    points = _extract_points(df)
    if points.empty or len(points) < 3:
        return pd.DataFrame()

    nodes = points["node"].tolist()
    coords = points[["x", "y"]].to_numpy(dtype=float)
    distances = _distance_matrix(coords)
    route = _nearest_neighbor_route(distances)
    route = _two_opt(route, distances)
    return _route_frame(nodes, distances, route, "nearest_neighbor_2opt_tsp")


def vrp_savings_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    points = _extract_points(df, require_demand=True)
    if points.empty or len(points) < 3:
        return pd.DataFrame()

    depot_idx = _depot_position(points)
    depot = points.iloc[depot_idx]
    customers = [i for i in range(len(points)) if i != depot_idx and float(points.iloc[i]["demand"]) > 0]
    if not customers:
        return pd.DataFrame()

    coords = points[["x", "y"]].to_numpy(dtype=float)
    distances = _distance_matrix(coords)
    demands = points["demand"].to_numpy(dtype=float)
    capacity_values = _numeric_column_values(df, ("capacity", "vehicle_capacity", "limit"))
    if capacity_values.size:
        vehicle_capacity = float(np.nanmax(capacity_values))
    else:
        vehicle_capacity = max(float(np.nanmax(demands)), float(demands[customers].sum() / max(1, math.ceil(len(customers) / 3))))
    if vehicle_capacity <= 0:
        return pd.DataFrame()

    routes: list[list[int]] = [[customer] for customer in customers if demands[customer] <= vehicle_capacity]
    if not routes:
        return pd.DataFrame()
    loads = [float(demands[route].sum()) for route in routes]
    savings = []
    for left, right in itertools.combinations(customers, 2):
        savings.append((distances[depot_idx, left] + distances[depot_idx, right] - distances[left, right], left, right))
    savings.sort(reverse=True)

    for _, left, right in savings:
        left_route_idx = _route_containing(routes, left)
        right_route_idx = _route_containing(routes, right)
        if left_route_idx is None or right_route_idx is None or left_route_idx == right_route_idx:
            continue
        left_route = routes[left_route_idx]
        right_route = routes[right_route_idx]
        if loads[left_route_idx] + loads[right_route_idx] > vehicle_capacity:
            continue
        merged = _merge_routes_on_ends(left_route, right_route, left, right)
        if merged is None:
            continue
        keep, drop = sorted((left_route_idx, right_route_idx))
        routes[keep] = merged
        loads[keep] = loads[left_route_idx] + loads[right_route_idx]
        del routes[drop]
        del loads[drop]

    rows = []
    node_names = points["node"].tolist()
    for vehicle_id, route in enumerate(routes, start=1):
        previous = depot_idx
        cumulative = 0.0
        route_load = float(demands[route].sum())
        full_route = [depot_idx, *route, depot_idx]
        for sequence, node_idx in enumerate(full_route):
            leg = 0.0 if sequence == 0 else float(distances[previous, node_idx])
            cumulative += leg
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "sequence": sequence,
                    "node": str(node_names[node_idx]),
                    "demand": float(demands[node_idx]),
                    "route_load": route_load,
                    "vehicle_capacity": vehicle_capacity,
                    "leg_distance": leg,
                    "cumulative_distance": cumulative,
                    "depot": str(depot["node"]),
                    "method": "clarke_wright_savings",
                }
            )
            previous = node_idx
    return pd.DataFrame(rows)


def dynamic_programming_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Solve a small staged resource-allocation problem with dynamic programming."""

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    stage_col = _find_column(df, ("stage", "period", "step", "time"))
    resource_col = _find_column(numeric, ("resource", "capacity", "budget", "amount", "allocation"))
    value_col = _find_column(numeric, BENEFIT_KEYWORDS + ("reward", "return", "utility"), exclude={resource_col})
    if resource_col is None:
        resource_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != resource_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

    work = pd.DataFrame(
        {
            "stage": df[stage_col].astype(str) if stage_col else [f"stage_{idx}" for idx in df.index],
            "resource": pd.to_numeric(numeric[resource_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
        }
    ).dropna()
    work = work[(work["resource"] >= 0) & (work["value"] >= 0)].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    total_capacity_values = _numeric_column_values(df, ("total_capacity", "total_budget", "capacity_limit"))
    if total_capacity_values.size:
        total_capacity = int(max(1, round(float(total_capacity_values[0]))))
    else:
        total_capacity = int(max(1, round(float(work["resource"].sum() * 0.5))))
    if total_capacity > 500:
        scale = total_capacity / 500
        work["resource"] = (work["resource"] / scale).round().clip(lower=0)
        total_capacity = 500

    costs = work["resource"].round().astype(int).to_numpy()
    values = work["value"].to_numpy(dtype=float)
    n = len(work)
    dp = np.zeros((n + 1, total_capacity + 1), dtype=float)
    take = np.zeros((n + 1, total_capacity + 1), dtype=bool)
    for i in range(1, n + 1):
        cost = int(costs[i - 1])
        value = float(values[i - 1])
        for capacity in range(total_capacity + 1):
            best = dp[i - 1, capacity]
            if cost <= capacity and dp[i - 1, capacity - cost] + value > best:
                dp[i, capacity] = dp[i - 1, capacity - cost] + value
                take[i, capacity] = True
            else:
                dp[i, capacity] = best

    selected: list[int] = []
    capacity = total_capacity
    for i in range(n, 0, -1):
        if take[i, capacity]:
            selected.append(i - 1)
            capacity -= int(costs[i - 1])
    selected.reverse()
    selected_set = set(selected)

    rows = []
    cumulative_resource = 0
    cumulative_value = 0.0
    for idx, row in work.iterrows():
        is_selected = idx in selected_set
        if is_selected:
            cumulative_resource += int(costs[idx])
            cumulative_value += float(values[idx])
        rows.append(
            {
                "stage": row["stage"],
                "selected": int(is_selected),
                "resource": int(costs[idx]),
                "value": float(values[idx]),
                "total_capacity": int(total_capacity),
                "cumulative_resource": int(cumulative_resource),
                "cumulative_value": float(cumulative_value),
                "optimal_value": float(dp[n, total_capacity]),
                "method": "dynamic_programming_resource_allocation",
            }
        )
    return pd.DataFrame(rows)


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    exclude = exclude or set()
    for column in df.columns:
        if str(column) in exclude:
            continue
        if _looks_like(str(column), keywords):
            return str(column)
    return None


def _looks_like(name: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(name).lower()
    return any(keyword == lowered or keyword in lowered for keyword in keywords)


def _quadratic_design(x: np.ndarray) -> np.ndarray:
    columns = [np.ones(x.shape[0])]
    columns.extend(x[:, i] for i in range(x.shape[1]))
    columns.extend(x[:, i] ** 2 for i in range(x.shape[1]))
    columns.extend(x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i + 1, x.shape[1]))
    return np.column_stack(columns)


def _quadratic_predict(z: np.ndarray, coef: np.ndarray) -> float:
    return float(_quadratic_design(z.reshape(1, -1))[0] @ coef)


def _quadratic_gradient(z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    p = len(z)
    grad = np.array(coef[1 : 1 + p], dtype=float)
    square_offset = 1 + p
    grad += 2.0 * coef[square_offset : square_offset + p] * z
    offset = square_offset + p
    for i in range(p):
        for j in range(i + 1, p):
            grad[i] += coef[offset] * z[j]
            grad[j] += coef[offset] * z[i]
            offset += 1
    return grad


def _branch_bound_knapsack(weights: np.ndarray, values: np.ndarray, capacity: float) -> list[int]:
    order = np.argsort(-(values / weights))
    sorted_weights = weights[order]
    sorted_values = values[order]
    best_value = 0.0
    best_taken: list[int] = []

    def bound(level: int, current_weight: float, current_value: float) -> float:
        remaining = capacity - current_weight
        estimate = current_value
        for idx in range(level, len(order)):
            if sorted_weights[idx] <= remaining:
                remaining -= sorted_weights[idx]
                estimate += sorted_values[idx]
            else:
                estimate += sorted_values[idx] * remaining / sorted_weights[idx]
                break
        return estimate

    stack: list[tuple[int, float, float, list[int]]] = [(0, 0.0, 0.0, [])]
    while stack:
        level, current_weight, current_value, taken = stack.pop()
        if current_value > best_value:
            best_value = current_value
            best_taken = taken
        if level >= len(order) or bound(level, current_weight, current_value) <= best_value + 1e-12:
            continue
        item = int(order[level])
        weight = float(sorted_weights[level])
        value = float(sorted_values[level])
        if current_weight + weight <= capacity:
            stack.append((level + 1, current_weight + weight, current_value + value, [*taken, item]))
        stack.append((level + 1, current_weight, current_value, taken))
    return sorted(best_taken)


def _greedy_knapsack(weights: np.ndarray, values: np.ndarray, capacity: float) -> list[int]:
    selected: list[int] = []
    used = 0.0
    for idx in np.argsort(-(values / weights)):
        if used + weights[idx] <= capacity:
            selected.append(int(idx))
            used += float(weights[idx])
    return sorted(selected)


def _criteria_weights(df: pd.DataFrame, criteria: list[str]) -> dict[str, float]:
    weights = {column: 1.0 for column in criteria}
    for column in df.columns:
        name = str(column).lower()
        if not name.startswith("weight"):
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        suffix = name.replace("weight", "", 1).strip("_ ")
        for criterion in criteria:
            if suffix and suffix in criterion.lower():
                weights[criterion] = float(values.iloc[0])
    return weights


def _extract_edges(df: pd.DataFrame) -> pd.DataFrame:
    source_col = _find_column(df, ("source", "from", "start", "origin"))
    target_col = _find_column(df, ("target", "to", "end", "dest", "destination"))
    if source_col is None or target_col is None:
        return pd.DataFrame()
    weight_col = _find_column(df, ("weight", "distance", "cost", "length", "time"))
    result = pd.DataFrame({"source": df[source_col].astype(str), "target": df[target_col].astype(str)})
    result["weight"] = pd.to_numeric(df[weight_col], errors="coerce") if weight_col else 1.0
    result = result.dropna()
    return result[result["weight"] > 0].reset_index(drop=True)


def _astar_edges(edges: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    nodes = sorted(set(edges["source"]) | set(edges["target"]))
    if len(nodes) < 2:
        return pd.DataFrame()
    start_col = _find_column(original, ("is_start", "start_flag"))
    end_col = _find_column(original, ("is_end", "end_flag", "goal_flag"))
    start = str(edges.iloc[0]["source"])
    goal = str(edges.iloc[-1]["target"])
    if start_col is not None:
        starts = original.loc[pd.to_numeric(original[start_col], errors="coerce").fillna(0) > 0]
        if not starts.empty:
            start = str(starts.iloc[0].get(_find_column(original, ("source", "from", "origin")) or starts.columns[0]))
    if end_col is not None:
        goals = original.loc[pd.to_numeric(original[end_col], errors="coerce").fillna(0) > 0]
        if not goals.empty:
            goal = str(goals.iloc[0].get(_find_column(original, ("target", "to", "destination")) or goals.columns[0]))

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        adjacency[str(row.source)].append((str(row.target), float(row.weight)))
        adjacency[str(row.target)].append((str(row.source), float(row.weight)))
    path, cost = _astar(adjacency, start, goal, lambda _: 0.0)
    return _path_result(path, cost, "astar_graph_edges")


def _extract_grid(df: pd.DataFrame) -> pd.DataFrame:
    x_col = _find_column(df, ("x", "col", "column", "longitude", "lon"))
    y_col = _find_column(df, ("y", "row", "latitude", "lat"))
    if x_col is None or y_col is None:
        return pd.DataFrame()
    grid = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")})
    obstacle_col = _find_column(df, ("obstacle", "blocked", "barrier", "wall"))
    cost_col = _find_column(df, ("cost", "weight", "distance", "time"))
    start_col = _find_column(df, ("start", "is_start"))
    goal_col = _find_column(df, ("goal", "end", "is_end", "target"))
    grid["blocked"] = pd.to_numeric(df[obstacle_col], errors="coerce").fillna(0) > 0 if obstacle_col else False
    grid["cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(1.0) if cost_col else 1.0
    grid["is_start"] = pd.to_numeric(df[start_col], errors="coerce").fillna(0) > 0 if start_col else False
    grid["is_goal"] = pd.to_numeric(df[goal_col], errors="coerce").fillna(0) > 0 if goal_col else False
    grid = grid.dropna(subset=["x", "y"])
    return grid[grid["cost"] > 0].reset_index(drop=True)


def _astar_grid(grid: pd.DataFrame) -> pd.DataFrame:
    free = grid[~grid["blocked"]].copy()
    if len(free) < 2:
        return pd.DataFrame()
    positions = {(int(row.x), int(row.y)): float(row.cost) for row in free.itertuples(index=False)}
    if free["is_start"].any():
        start_row = free[free["is_start"]].iloc[0]
    else:
        start_row = free.assign(score=free["x"] + free["y"]).sort_values("score").iloc[0]
    if free["is_goal"].any():
        goal_row = free[free["is_goal"]].iloc[0]
    else:
        goal_row = free.assign(score=free["x"] + free["y"]).sort_values("score").iloc[-1]
    start = (int(start_row["x"]), int(start_row["y"]))
    goal = (int(goal_row["x"]), int(goal_row["y"]))

    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = defaultdict(list)
    for x, y in positions:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (x + dx, y + dy)
            if neighbor in positions:
                adjacency[(x, y)].append((neighbor, positions[neighbor]))
    heuristic = lambda node: float(abs(node[0] - goal[0]) + abs(node[1] - goal[1]))
    path, cost = _astar(adjacency, start, goal, heuristic)
    if not path:
        return pd.DataFrame()
    rows = []
    cumulative = 0.0
    previous = None
    for sequence, node in enumerate(path):
        if previous is not None:
            cumulative += positions[node]
        rows.append(
            {
                "sequence": sequence,
                "node": f"{node[0]},{node[1]}",
                "x": node[0],
                "y": node[1],
                "step_cost": 0.0 if previous is None else positions[node],
                "cumulative_cost": cumulative,
                "total_cost": cost,
                "method": "astar_grid_manhattan",
            }
        )
        previous = node
    return pd.DataFrame(rows)


def _astar(adjacency, start, goal, heuristic):
    queue = [(heuristic(start), 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    while queue:
        _, cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost > cost_so_far.get(node, float("inf")):
            continue
        for neighbor, weight in adjacency.get(node, []):
            new_cost = cost + float(weight)
            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(queue, (new_cost + heuristic(neighbor), new_cost, neighbor))
    if goal not in came_from:
        return [], float("inf")
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, float(cost_so_far[goal])


def _path_result(path: list[str], total_cost: float, method: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    rows = []
    for sequence, node in enumerate(path):
        rows.append({"sequence": sequence, "node": str(node), "total_cost": total_cost, "method": method})
    return pd.DataFrame(rows)


def _extract_points(df: pd.DataFrame, require_demand: bool = False) -> pd.DataFrame:
    x_col = _find_column(df, ("x", "longitude", "lon"))
    y_col = _find_column(df, ("y", "latitude", "lat"))
    if x_col is None or y_col is None:
        numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if numeric.shape[1] < 2:
            return pd.DataFrame()
        x_col, y_col = str(numeric.columns[0]), str(numeric.columns[1])
    node_col = _find_column(df, ("node", "city", "customer", "location", "name"))
    demand_col = _find_column(df, ("demand", "load", "quantity", "volume"))
    points = pd.DataFrame(
        {
            "node": df[node_col].astype(str) if node_col else [f"node_{idx}" for idx in df.index],
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "demand": pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0) if demand_col else 0.0,
        }
    ).dropna(subset=["x", "y"])
    if require_demand and points["demand"].sum() <= 0:
        return pd.DataFrame()
    return points.reset_index(drop=True)


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((delta**2).sum(axis=2))


def _nearest_neighbor_route(distances: np.ndarray) -> list[int]:
    route = [0]
    unvisited = set(range(1, len(distances)))
    while unvisited:
        current = route[-1]
        next_node = min(unvisited, key=lambda node: distances[current, node])
        route.append(next_node)
        unvisited.remove(next_node)
    return route


def _two_opt(route: list[int], distances: np.ndarray) -> list[int]:
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                if _route_distance(candidate, distances) + 1e-12 < _route_distance(best, distances):
                    best = candidate
                    improved = True
        route = best
    return best


def _route_distance(route: list[int], distances: np.ndarray) -> float:
    total = 0.0
    for left, right in zip(route, route[1:]):
        total += float(distances[left, right])
    total += float(distances[route[-1], route[0]])
    return total


def _route_frame(nodes: list[str], distances: np.ndarray, route: list[int], method: str) -> pd.DataFrame:
    total = _route_distance(route, distances)
    rows = []
    cumulative = 0.0
    cycle = [*route, route[0]]
    for sequence, (left, right) in enumerate(zip(cycle, cycle[1:]), start=1):
        leg = float(distances[left, right])
        cumulative += leg
        rows.append(
            {
                "sequence": sequence,
                "node": str(nodes[left]),
                "next_node": str(nodes[right]),
                "leg_distance": leg,
                "cumulative_distance": cumulative,
                "total_distance": total,
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def _depot_position(points: pd.DataFrame) -> int:
    for idx, node in enumerate(points["node"].astype(str)):
        if "depot" in node.lower() or "warehouse" in node.lower():
            return idx
    zero_demand = points.index[points["demand"] <= 0]
    return int(zero_demand[0]) if len(zero_demand) else 0


def _numeric_column_values(df: pd.DataFrame, keywords: tuple[str, ...]) -> np.ndarray:
    column = _find_column(df, keywords)
    if column is None:
        return np.array([])
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)


def _route_containing(routes: list[list[int]], node: int) -> int | None:
    for idx, route in enumerate(routes):
        if node in route:
            return idx
    return None


def _merge_routes_on_ends(left_route: list[int], right_route: list[int], left: int, right: int) -> list[int] | None:
    variants = [
        (left_route, right_route),
        (left_route[::-1], right_route),
        (left_route, right_route[::-1]),
        (left_route[::-1], right_route[::-1]),
    ]
    for first, second in variants:
        if first[-1] == left and second[0] == right:
            return [*first, *second]
        if first[-1] == right and second[0] == left:
            return [*first, *second]
    return None
