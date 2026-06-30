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


def robust_resource_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative resource-allocation plan under row-level uncertainty.

    The routine is intentionally solver-light: it identifies a benefit column,
    a resource/cost column, and an optional uncertainty/risk column, then selects
    alternatives by robust benefit density while enforcing an inflated resource
    capacity. This gives the workflow an executable robust-optimization artifact
    even when the contest statement does not provide a fully specified stochastic
    program.
    """

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    resource_col = _find_column(
        numeric,
        ("resource", "capacity_used", "weight", "cost", "area", "time", "budget_used", "load"),
    )
    value_col = _find_column(
        numeric,
        BENEFIT_KEYWORDS + ("reward", "margin", "objective"),
        exclude={resource_col},
    )
    uncertainty_col = _find_column(
        numeric,
        ("uncertainty", "risk", "std", "sigma", "variance", "volatility", "error", "loss_rate", "penalty"),
        exclude={resource_col, value_col},
    )
    capacity_col = _find_column(
        numeric,
        ("capacity", "budget", "limit", "available", "bound"),
        exclude={resource_col, value_col, uncertainty_col},
    )
    item_col = _find_column(df, ("item", "name", "crop", "project", "task", "option", "node", "id"))

    if resource_col is None:
        resource_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != resource_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

    work = pd.DataFrame(
        {
            "row_index": numeric.index,
            "item": df[item_col].astype(str) if item_col else [f"option_{idx}" for idx in numeric.index],
            "resource": pd.to_numeric(numeric[resource_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
        }
    )
    if uncertainty_col is not None:
        uncertainty = pd.to_numeric(numeric[uncertainty_col], errors="coerce")
        scale = float(uncertainty.abs().quantile(0.9)) if not uncertainty.dropna().empty else 0.0
        if scale > 1.0:
            uncertainty = uncertainty / scale
        work["uncertainty"] = uncertainty.abs().clip(lower=0.0, upper=1.0)
    else:
        value_series = pd.to_numeric(numeric[value_col], errors="coerce")
        coefficient = float(value_series.std() / max(abs(value_series.mean()), 1e-9)) if value_series.notna().sum() >= 2 else 0.1
        work["uncertainty"] = min(max(coefficient, 0.05), 0.35)

    work = work.dropna(subset=["resource", "value", "uncertainty"])
    work = work[(work["resource"] > 0) & (work["value"] > 0)].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        capacity = float(work["resource"].sum() * 0.5)
    if capacity <= 0:
        return pd.DataFrame()

    risk_aversion = 1.0
    budget_buffer = 0.2
    work["robust_resource"] = work["resource"] * (1.0 + budget_buffer * work["uncertainty"])
    work["robust_value"] = work["value"] * (1.0 - risk_aversion * work["uncertainty"]).clip(lower=0.0)
    work["robust_density"] = work["robust_value"] / work["robust_resource"]
    ranked = work.sort_values(["robust_density", "robust_value"], ascending=False).reset_index(drop=True)

    selected: list[int] = []
    used_resource = 0.0
    for idx, row in ranked.iterrows():
        candidate_resource = float(row["robust_resource"])
        if used_resource + candidate_resource <= capacity + 1e-12:
            selected.append(idx)
            used_resource += candidate_resource

    selected_set = set(selected)
    selected_rows = ranked.loc[list(selected_set)] if selected_set else ranked.iloc[0:0]
    total_value = float(selected_rows["value"].sum())
    total_robust_value = float(selected_rows["robust_value"].sum())
    rows = []
    for idx, row in ranked.iterrows():
        is_selected = idx in selected_set
        rows.append(
            {
                "row_index": int(row["row_index"]),
                "item": str(row["item"]),
                "selected": int(is_selected),
                "resource_column": str(resource_col),
                "value_column": str(value_col),
                "uncertainty_column": str(uncertainty_col) if uncertainty_col else "estimated_cv",
                "nominal_resource": float(row["resource"]),
                "robust_resource": float(row["robust_resource"]),
                "nominal_value": float(row["value"]),
                "robust_value": float(row["robust_value"]),
                "uncertainty_rate": float(row["uncertainty"]),
                "capacity": capacity,
                "total_robust_resource": float(used_resource),
                "capacity_slack": float(capacity - used_resource),
                "total_nominal_value": float(total_value),
                "total_robust_value": float(total_robust_value),
                "method": "greedy_budgeted_robust_optimization",
            }
        )
    return pd.DataFrame(rows)


def scenario_resource_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate resource decisions across explicit or generated scenarios."""

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    item_col = _find_column(df, ("item", "name", "crop", "project", "task", "option", "node", "id"))
    scenario_col = _find_column(df, ("scenario", "case", "state", "condition"))
    resource_col = _find_column(
        numeric,
        ("resource", "capacity_used", "weight", "cost", "area", "time", "budget_used", "load"),
    )
    value_col = _find_column(
        numeric,
        BENEFIT_KEYWORDS + ("reward", "margin", "objective"),
        exclude={resource_col},
    )
    capacity_col = _find_column(
        numeric,
        ("capacity", "budget", "limit", "available", "bound"),
        exclude={resource_col, value_col},
    )
    probability_col = _find_column(
        numeric,
        ("probability", "prob", "likelihood", "chance", "weight_scenario"),
        exclude={resource_col, value_col, capacity_col},
    )
    if resource_col is None:
        resource_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != resource_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

    base = pd.DataFrame(
        {
            "row_index": numeric.index,
            "item": df[item_col].astype(str) if item_col else [f"option_{idx}" for idx in numeric.index],
            "scenario": df[scenario_col].astype(str) if scenario_col else "base",
            "resource": pd.to_numeric(numeric[resource_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
            "probability": pd.to_numeric(numeric[probability_col], errors="coerce") if probability_col else np.nan,
        }
    ).dropna(subset=["resource", "value"])
    base = base[(base["resource"] > 0) & (base["value"] > 0)].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        nominal = base.drop_duplicates("item")
        capacity = float(nominal["resource"].sum() * 0.5)
    if capacity <= 0:
        return pd.DataFrame()

    scenarios = _scenario_table(base, explicit=bool(scenario_col))
    if scenarios.empty:
        return pd.DataFrame()

    summary = _scenario_item_summary(scenarios)
    if summary.empty:
        return pd.DataFrame()

    summary["risk_adjusted_score"] = (
        0.55 * summary["expected_value"]
        + 0.35 * summary["worst_case_value"]
        - 0.10 * summary["max_regret"]
    )
    summary["density"] = summary["risk_adjusted_score"] / summary["mean_resource"].clip(lower=1e-9)
    ranked = summary.sort_values(["density", "worst_case_value"], ascending=False).reset_index(drop=True)

    selected: list[int] = []
    used_resource = 0.0
    for idx, row in ranked.iterrows():
        resource = float(row["mean_resource"])
        if used_resource + resource <= capacity + 1e-12:
            selected.append(idx)
            used_resource += resource
    selected_set = set(selected)

    rows: list[dict[str, float | int | str]] = []
    for idx, row in ranked.iterrows():
        item = str(row["item"])
        item_scenarios = scenarios[scenarios["item"] == item].sort_values("scenario")
        for scenario_row in item_scenarios.itertuples(index=False):
            rows.append(
                {
                    "item": item,
                    "scenario": str(scenario_row.scenario),
                    "selected": int(idx in selected_set),
                    "scenario_probability": float(scenario_row.probability),
                    "resource": float(scenario_row.resource),
                    "scenario_value": float(scenario_row.value),
                    "expected_value": float(row["expected_value"]),
                    "worst_case_value": float(row["worst_case_value"]),
                    "best_case_value": float(row["best_case_value"]),
                    "max_regret": float(row["max_regret"]),
                    "risk_adjusted_score": float(row["risk_adjusted_score"]),
                    "capacity": capacity,
                    "total_selected_resource": float(used_resource),
                    "capacity_slack": float(capacity - used_resource),
                    "method": "scenario_expected_worstcase_regret_optimization",
                }
            )
    return pd.DataFrame(rows)


def chance_constrained_resource_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Select decisions that satisfy a resource chance constraint.

    The model estimates an uncertain resource requirement for each option and
    inflates it by a service-level quantile. It then selects options by safe
    value density and reports feasibility/violation probabilities.
    """

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    item_col = _find_column(df, ("item", "name", "crop", "project", "task", "option", "node", "id"))
    resource_col = _find_column(
        numeric,
        ("resource", "demand", "capacity_used", "weight", "load", "area", "time", "budget_used"),
    )
    value_col = _find_column(
        numeric,
        BENEFIT_KEYWORDS + ("reward", "margin", "objective"),
        exclude={resource_col},
    )
    std_col = _find_column(
        numeric,
        ("std", "sigma", "uncertainty", "deviation", "volatility", "risk"),
        exclude={resource_col, value_col},
    )
    capacity_col = _find_column(
        numeric,
        ("capacity", "budget", "limit", "available", "bound", "supply"),
        exclude={resource_col, value_col, std_col},
    )
    service_level_col = _find_column(
        numeric,
        ("service_level", "confidence", "reliability", "probability"),
        exclude={resource_col, value_col, std_col, capacity_col},
    )
    if resource_col is None:
        resource_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != resource_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

    work = pd.DataFrame(
        {
            "row_index": numeric.index,
            "item": df[item_col].astype(str) if item_col else [f"option_{idx}" for idx in numeric.index],
            "mean_resource": pd.to_numeric(numeric[resource_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
        }
    )
    if std_col is not None:
        work["resource_std"] = pd.to_numeric(numeric[std_col], errors="coerce").abs()
    else:
        resource_series = pd.to_numeric(numeric[resource_col], errors="coerce")
        coefficient = float(resource_series.std() / max(abs(resource_series.mean()), 1e-9)) if resource_series.notna().sum() >= 2 else 0.15
        work["resource_std"] = pd.to_numeric(work["mean_resource"], errors="coerce").abs() * min(max(coefficient, 0.05), 0.35)
    work = work.dropna(subset=["mean_resource", "value", "resource_std"])
    work = work[(work["mean_resource"] > 0) & (work["value"] > 0)].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        capacity = float(work["mean_resource"].sum() * 0.5)
    if capacity <= 0:
        return pd.DataFrame()

    if service_level_col is not None:
        service_values = pd.to_numeric(numeric[service_level_col], errors="coerce").dropna()
        service_level = float(service_values.iloc[0]) if not service_values.empty else 0.9
    else:
        service_level = 0.9
    if service_level > 1.0:
        service_level /= 100.0
    service_level = min(max(service_level, 0.5), 0.99)
    z_value = _service_level_z(service_level)

    work["safe_resource"] = work["mean_resource"] + z_value * work["resource_std"]
    work["safe_density"] = work["value"] / work["safe_resource"].clip(lower=1e-9)
    ranked = work.sort_values(["safe_density", "value"], ascending=False).reset_index(drop=True)

    selected: list[int] = []
    mean_total = 0.0
    variance_total = 0.0
    safe_total = 0.0
    for idx, row in ranked.iterrows():
        candidate_safe_total = safe_total + float(row["safe_resource"])
        if candidate_safe_total <= capacity + 1e-12:
            selected.append(idx)
            mean_total += float(row["mean_resource"])
            variance_total += float(row["resource_std"]) ** 2
            safe_total = candidate_safe_total
    selected_set = set(selected)
    total_std = math.sqrt(max(variance_total, 0.0))
    feasibility_probability = _normal_cdf((capacity - mean_total) / total_std) if total_std > 0 else float(mean_total <= capacity)
    violation_probability = 1.0 - feasibility_probability
    total_value = float(ranked.loc[list(selected_set), "value"].sum()) if selected_set else 0.0

    rows = []
    for idx, row in ranked.iterrows():
        rows.append(
            {
                "row_index": int(row["row_index"]),
                "item": str(row["item"]),
                "selected": int(idx in selected_set),
                "mean_resource": float(row["mean_resource"]),
                "resource_std": float(row["resource_std"]),
                "safe_resource": float(row["safe_resource"]),
                "value": float(row["value"]),
                "safe_density": float(row["safe_density"]),
                "service_level": service_level,
                "z_value": z_value,
                "capacity": capacity,
                "total_mean_resource": float(mean_total),
                "total_safe_resource": float(safe_total),
                "capacity_slack": float(capacity - safe_total),
                "feasibility_probability": float(feasibility_probability),
                "violation_probability": float(violation_probability),
                "total_value": total_value,
                "method": "chance_constrained_resource_optimization",
            }
        )
    return pd.DataFrame(rows)


def cvar_resource_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize resource selection with CVaR-style downside risk control."""

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    item_col = _find_column(df, ("item", "name", "crop", "project", "task", "option", "node", "id"))
    scenario_col = _find_column(df, ("scenario", "case", "state", "condition"))
    resource_col = _find_column(
        numeric,
        ("resource", "capacity_used", "weight", "cost", "area", "time", "budget_used", "load"),
    )
    value_col = _find_column(
        numeric,
        BENEFIT_KEYWORDS + ("reward", "margin", "objective"),
        exclude={resource_col},
    )
    risk_col = _find_column(
        numeric,
        ("loss", "risk", "uncertainty", "std", "sigma", "volatility", "penalty", "shortage"),
        exclude={resource_col, value_col},
    )
    capacity_col = _find_column(
        numeric,
        ("capacity", "budget", "limit", "available", "bound"),
        exclude={resource_col, value_col, risk_col},
    )
    alpha_col = _find_column(
        numeric,
        ("alpha", "confidence", "cvar_level", "risk_level"),
        exclude={resource_col, value_col, risk_col, capacity_col},
    )
    if resource_col is None:
        resource_col = str(numeric.columns[0])
    if value_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != resource_col]
        if not candidates:
            return pd.DataFrame()
        value_col = candidates[-1]

    base = pd.DataFrame(
        {
            "row_index": numeric.index,
            "item": df[item_col].astype(str) if item_col else [f"option_{idx}" for idx in numeric.index],
            "scenario": df[scenario_col].astype(str) if scenario_col else "base",
            "resource": pd.to_numeric(numeric[resource_col], errors="coerce"),
            "value": pd.to_numeric(numeric[value_col], errors="coerce"),
        }
    )
    if risk_col is not None:
        base["risk"] = pd.to_numeric(numeric[risk_col], errors="coerce").abs()
    else:
        value_series = pd.to_numeric(numeric[value_col], errors="coerce")
        coefficient = float(value_series.std() / max(abs(value_series.mean()), 1e-9)) if value_series.notna().sum() >= 2 else 0.15
        base["risk"] = pd.to_numeric(base["value"], errors="coerce").abs() * min(max(coefficient, 0.05), 0.35)
    base = base.dropna(subset=["resource", "value", "risk"])
    base = base[(base["resource"] > 0) & (base["value"] > 0)].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame()

    if capacity_col is not None:
        capacity_values = pd.to_numeric(numeric[capacity_col], errors="coerce").dropna()
        capacity = float(capacity_values.iloc[0]) if not capacity_values.empty else 0.0
    else:
        capacity = float(base.drop_duplicates("item")["resource"].sum() * 0.5)
    if capacity <= 0:
        return pd.DataFrame()

    if alpha_col is not None:
        alpha_values = pd.to_numeric(numeric[alpha_col], errors="coerce").dropna()
        alpha = float(alpha_values.iloc[0]) if not alpha_values.empty else 0.9
    else:
        alpha = 0.9
    if alpha > 1.0:
        alpha /= 100.0
    alpha = min(max(alpha, 0.5), 0.99)

    scenarios = _cvar_scenario_table(base, explicit=bool(scenario_col))
    summary = _cvar_item_summary(scenarios, alpha=alpha)
    if summary.empty:
        return pd.DataFrame()
    risk_aversion = 0.65
    summary["risk_adjusted_score"] = summary["expected_value"] - risk_aversion * summary["cvar_loss"]
    summary["risk_adjusted_density"] = summary["risk_adjusted_score"] / summary["mean_resource"].clip(lower=1e-9)
    ranked = summary.sort_values(["risk_adjusted_density", "expected_value"], ascending=False).reset_index(drop=True)

    selected: list[int] = []
    used_resource = 0.0
    for idx, row in ranked.iterrows():
        resource = float(row["mean_resource"])
        if used_resource + resource <= capacity + 1e-12:
            selected.append(idx)
            used_resource += resource
    selected_set = set(selected)
    total_expected_value = float(ranked.loc[list(selected_set), "expected_value"].sum()) if selected_set else 0.0
    total_cvar_loss = float(ranked.loc[list(selected_set), "cvar_loss"].sum()) if selected_set else 0.0

    rows: list[dict[str, float | int | str]] = []
    for idx, row in ranked.iterrows():
        rows.append(
            {
                "item": str(row["item"]),
                "selected": int(idx in selected_set),
                "mean_resource": float(row["mean_resource"]),
                "expected_value": float(row["expected_value"]),
                "worst_case_value": float(row["worst_case_value"]),
                "var_loss": float(row["var_loss"]),
                "cvar_loss": float(row["cvar_loss"]),
                "tail_scenario_count": int(row["tail_scenario_count"]),
                "risk_adjusted_score": float(row["risk_adjusted_score"]),
                "risk_adjusted_density": float(row["risk_adjusted_density"]),
                "confidence_level": alpha,
                "capacity": capacity,
                "total_selected_resource": float(used_resource),
                "capacity_slack": float(capacity - used_resource),
                "total_expected_value": total_expected_value,
                "total_cvar_loss": total_cvar_loss,
                "method": "cvar_tail_risk_resource_optimization",
            }
        )
    return pd.DataFrame(rows)


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


def _scenario_table(base: pd.DataFrame, *, explicit: bool) -> pd.DataFrame:
    if explicit:
        scenarios = base.copy()
        probability = pd.to_numeric(scenarios["probability"], errors="coerce")
        if probability.notna().sum() == 0:
            counts = scenarios.groupby("scenario")["scenario"].transform("count")
            scenario_count = max(int(scenarios["scenario"].nunique()), 1)
            scenarios["probability"] = 1.0 / scenario_count / counts
        else:
            scenarios["probability"] = probability.fillna(0.0)
            total = float(scenarios["probability"].sum())
            scenarios["probability"] = scenarios["probability"] / total if total > 0 else 1.0 / len(scenarios)
        return scenarios

    rows: list[dict[str, float | str]] = []
    for row in base.itertuples(index=False):
        for scenario, multiplier, probability in (
            ("pessimistic", 0.8, 0.25),
            ("base", 1.0, 0.50),
            ("optimistic", 1.15, 0.25),
        ):
            rows.append(
                {
                    "item": str(row.item),
                    "scenario": scenario,
                    "resource": float(row.resource),
                    "value": float(row.value) * multiplier,
                    "probability": probability,
                }
            )
    return pd.DataFrame(rows)


def _scenario_item_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    scenario_best = scenarios.groupby("scenario")["value"].max().to_dict()
    rows = []
    for item, group in scenarios.groupby("item"):
        probabilities = pd.to_numeric(group["probability"], errors="coerce").fillna(0.0)
        total_probability = float(probabilities.sum())
        if total_probability <= 0:
            probabilities = pd.Series(np.ones(len(group)) / len(group), index=group.index)
        else:
            probabilities = probabilities / total_probability
        values = pd.to_numeric(group["value"], errors="coerce").fillna(0.0)
        resources = pd.to_numeric(group["resource"], errors="coerce").fillna(0.0)
        regrets = [
            float(scenario_best.get(str(scenario), value) - value)
            for scenario, value in zip(group["scenario"], values)
        ]
        rows.append(
            {
                "item": str(item),
                "expected_value": float((values * probabilities).sum()),
                "worst_case_value": float(values.min()),
                "best_case_value": float(values.max()),
                "max_regret": float(max(regrets) if regrets else 0.0),
                "mean_resource": float(resources.mean()),
            }
        )
    return pd.DataFrame(rows)


def _cvar_scenario_table(base: pd.DataFrame, *, explicit: bool) -> pd.DataFrame:
    if explicit:
        scenarios = base.copy()
        scenarios["loss"] = (scenarios["value"].max() - scenarios["value"] + scenarios["risk"]).clip(lower=0.0)
        return scenarios

    rows: list[dict[str, float | str]] = []
    for row in base.itertuples(index=False):
        for scenario, value_multiplier, risk_multiplier in (
            ("normal", 1.0, 1.0),
            ("stress", 0.85, 1.4),
            ("tail", 0.65, 2.0),
        ):
            value = float(row.value) * value_multiplier
            risk = float(row.risk) * risk_multiplier
            rows.append(
                {
                    "item": str(row.item),
                    "scenario": scenario,
                    "resource": float(row.resource),
                    "value": value,
                    "risk": risk,
                    "loss": max(0.0, float(row.value) - value + risk),
                }
            )
    return pd.DataFrame(rows)


def _cvar_item_summary(scenarios: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    rows = []
    for item, group in scenarios.groupby("item"):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        losses = pd.to_numeric(group["loss"], errors="coerce").dropna()
        resources = pd.to_numeric(group["resource"], errors="coerce").dropna()
        if values.empty or losses.empty or resources.empty:
            continue
        var_loss = float(losses.quantile(alpha, interpolation="higher"))
        tail = losses[losses >= var_loss]
        rows.append(
            {
                "item": str(item),
                "mean_resource": float(resources.mean()),
                "expected_value": float(values.mean()),
                "worst_case_value": float(values.min()),
                "var_loss": var_loss,
                "cvar_loss": float(tail.mean()) if not tail.empty else var_loss,
                "tail_scenario_count": int(len(tail)),
            }
        )
    return pd.DataFrame(rows)


def _service_level_z(service_level: float) -> float:
    table = (
        (0.50, 0.0),
        (0.80, 0.8416),
        (0.85, 1.0364),
        (0.90, 1.2816),
        (0.95, 1.6449),
        (0.975, 1.96),
        (0.99, 2.3263),
    )
    best_level, best_z = min(table, key=lambda item: abs(item[0] - service_level))
    if abs(best_level - service_level) <= 0.015:
        return best_z
    lower = max((item for item in table if item[0] <= service_level), default=table[0])
    upper = min((item for item in table if item[0] >= service_level), default=table[-1])
    if lower[0] == upper[0]:
        return lower[1]
    weight = (service_level - lower[0]) / (upper[0] - lower[0])
    return float(lower[1] + weight * (upper[1] - lower[1]))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))


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
