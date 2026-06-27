from __future__ import annotations

import math

import pandas as pd


def queue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    arrival_col = _find_column(df, ("arrival", "lambda", "到达", "到达率"))
    service_col = _find_column(df, ("service", "mu", "服务", "服务率"))
    servers_col = _find_column(df, ("server", "servers", "c", "窗口", "服务台"))
    if arrival_col is None or service_col is None:
        return pd.DataFrame()

    rows = []
    for idx, row in df.iterrows():
        arrival = _to_float(row.get(arrival_col))
        service = _to_float(row.get(service_col))
        servers = int(_to_float(row.get(servers_col), 1.0)) if servers_col else 1
        if arrival <= 0 or service <= 0 or servers <= 0:
            continue
        metrics = _mmc_metrics(arrival, service, servers)
        if metrics:
            metrics.update({"row_index": idx, "arrival_rate": arrival, "service_rate": service, "servers": servers})
            rows.append(metrics)
    return pd.DataFrame(rows)


def _mmc_metrics(arrival: float, service: float, servers: int) -> dict[str, float] | None:
    rho = arrival / (servers * service)
    if rho >= 1:
        return {"utilization": rho, "stable": 0.0}
    a = arrival / service
    sum_terms = sum((a**n) / math.factorial(n) for n in range(servers))
    last = (a**servers) / (math.factorial(servers) * (1 - rho))
    p0 = 1 / (sum_terms + last)
    lq = p0 * (a**servers) * rho / (math.factorial(servers) * (1 - rho) ** 2)
    wq = lq / arrival
    w = wq + 1 / service
    l = arrival * w
    return {"utilization": rho, "p0": p0, "lq": lq, "wq": wq, "l": l, "w": w, "stable": 1.0}


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any((keyword == name if len(keyword) <= 1 else keyword == name or keyword in name) for keyword in keywords):
            return str(column)
    return None


def _to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
