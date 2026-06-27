from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd


def multi_echelon_inventory(df: pd.DataFrame) -> pd.DataFrame:
    demand_col = _find_column(df, ("demand", "sales", "usage"))
    lead_time_col = _find_column(df, ("lead_time", "leadtime", "delay"))
    std_col = _find_column(df, ("demand_std", "std", "sigma", "volatility"))
    service_col = _find_column(df, ("service_level", "service", "fill_rate"))
    holding_col = _find_column(df, ("holding", "holding_cost", "storage"))
    order_col = _find_column(df, ("order_cost", "ordering_cost", "setup"))
    echelon_col = _find_column(df, ("echelon", "stage", "level", "node"))
    if demand_col is None:
        return pd.DataFrame()

    rows = []
    cumulative_reorder_point = 0.0
    for idx, row in df.iterrows():
        demand = _to_float(row.get(demand_col))
        lead_time = _to_float(row.get(lead_time_col), 1.0) if lead_time_col else 1.0
        service_level = _to_float(row.get(service_col), 0.95) if service_col else 0.95
        demand_std = _to_float(row.get(std_col), math.sqrt(max(demand, 0.0))) if std_col else math.sqrt(max(demand, 0.0))
        holding_cost = _to_float(row.get(holding_col), 1.0) if holding_col else 1.0
        order_cost = _to_float(row.get(order_col), 1.0) if order_col else 1.0
        if demand <= 0 or lead_time < 0:
            continue

        z_value = _service_z(service_level)
        safety_stock = z_value * demand_std * math.sqrt(lead_time)
        cycle_stock = demand * lead_time
        reorder_point = cycle_stock + safety_stock
        cumulative_reorder_point += reorder_point
        eoq = math.sqrt(2.0 * demand * order_cost / holding_cost) if order_cost > 0 and holding_cost > 0 else 0.0
        echelon = row.get(echelon_col) if echelon_col else idx
        rows.append(
            {
                "row_index": idx,
                "echelon": str(echelon),
                "demand": demand,
                "lead_time": lead_time,
                "service_level": service_level,
                "safety_stock": safety_stock,
                "cycle_stock": cycle_stock,
                "reorder_point": reorder_point,
                "echelon_inventory_position": cumulative_reorder_point,
                "eoq": eoq,
                "method": "multi_echelon_inventory",
            }
        )
    return pd.DataFrame(rows)


def bullwhip_effect(df: pd.DataFrame) -> pd.DataFrame:
    demand_col = _find_column(df, ("demand", "sales", "customer"))
    order_col = _find_column(df, ("order", "replenishment", "purchase"))
    echelon_col = _find_column(df, ("echelon", "stage", "level", "node"))
    if demand_col is None or order_col is None:
        return pd.DataFrame()

    groups = df.groupby(echelon_col, dropna=False) if echelon_col else [(None, df)]
    rows = []
    for echelon, group in groups:
        demand = pd.to_numeric(group[demand_col], errors="coerce").dropna()
        orders = pd.to_numeric(group[order_col], errors="coerce").dropna()
        aligned = pd.concat([demand, orders], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            continue
        demand_variance = float(aligned.iloc[:, 0].var(ddof=1))
        order_variance = float(aligned.iloc[:, 1].var(ddof=1))
        if demand_variance <= 0:
            ratio = math.inf if order_variance > 0 else 0.0
        else:
            ratio = order_variance / demand_variance
        rows.append(
            {
                "echelon": "" if echelon is None else str(echelon),
                "sample_size": int(len(aligned)),
                "demand_variance": demand_variance,
                "order_variance": order_variance,
                "bullwhip_ratio": ratio,
                "amplified": float(ratio > 1.0),
                "method": "bullwhip_effect",
            }
        )
    return pd.DataFrame(rows)


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None


def _service_z(service_level: float) -> float:
    clipped = min(max(service_level, 0.5), 0.999)
    return NormalDist().inv_cdf(clipped)


def _to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
