from __future__ import annotations

import math

import numpy as np
import pandas as pd


def jackson_network_queue(df: pd.DataFrame) -> pd.DataFrame:
    node_col = _find_column(df, ("node", "station", "queue", "server", "name"))
    arrival_col = _find_column(df, ("external_arrival", "arrival", "lambda"))
    service_col = _find_column(df, ("service", "mu"))
    servers_col = _find_column(df, ("servers", "server_count", "c"))
    if arrival_col is None or service_col is None:
        return pd.DataFrame()

    nodes = [str(value) for value in df[node_col]] if node_col else [str(i) for i in range(len(df))]
    if not nodes or len(set(nodes)) != len(nodes):
        return pd.DataFrame()

    external = np.array([_to_float(value) for value in df[arrival_col]], dtype=float)
    service = np.array([_to_float(value) for value in df[service_col]], dtype=float)
    servers = np.array(
        [_positive_int(_to_float(value, 1.0)) for value in df[servers_col]],
        dtype=int,
    ) if servers_col else np.ones(len(nodes), dtype=int)
    if np.any(external < 0) or np.any(service <= 0) or np.any(servers <= 0):
        return pd.DataFrame()

    routing = _routing_matrix(df, nodes)
    try:
        effective = np.linalg.solve(np.eye(len(nodes)) - routing.T, external)
    except np.linalg.LinAlgError:
        return pd.DataFrame()
    if np.any(effective < -1e-9):
        return pd.DataFrame()
    effective = np.maximum(effective, 0.0)

    rows = []
    for idx, node in enumerate(nodes):
        metrics = _mmc_metrics(float(effective[idx]), float(service[idx]), int(servers[idx]))
        metrics.update(
            {
                "node": node,
                "external_arrival_rate": float(external[idx]),
                "effective_arrival_rate": float(effective[idx]),
                "service_rate": float(service[idx]),
                "servers": int(servers[idx]),
                "routing_out_probability": float(routing[idx].sum()),
                "method": "jackson_network_queue",
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _routing_matrix(df: pd.DataFrame, nodes: list[str]) -> np.ndarray:
    routing = np.zeros((len(nodes), len(nodes)), dtype=float)
    node_index = {node.lower(): index for index, node in enumerate(nodes)}
    for i, _node in enumerate(nodes):
        for column in df.columns:
            name = str(column).lower()
            target = None
            for prefix in ("route_to_", "p_to_", "to_"):
                if name.startswith(prefix):
                    target = name[len(prefix) :]
                    break
            if target is None:
                continue
            j = node_index.get(target.lower())
            if j is not None:
                routing[i, j] = max(0.0, _to_float(df.iloc[i][column]))
    row_sums = routing.sum(axis=1)
    for i, total in enumerate(row_sums):
        if total > 1.0:
            routing[i] = routing[i] / total
    return routing


def _mmc_metrics(arrival: float, service: float, servers: int) -> dict[str, float]:
    if arrival == 0:
        return {"utilization": 0.0, "p0": 1.0, "lq": 0.0, "wq": 0.0, "l": 0.0, "w": 1.0 / service, "stable": 1.0}
    rho = arrival / (servers * service)
    if rho >= 1.0:
        return {"utilization": rho, "p0": 0.0, "lq": np.nan, "wq": np.nan, "l": np.nan, "w": np.nan, "stable": 0.0}
    a = arrival / service
    sum_terms = sum((a**n) / math.factorial(n) for n in range(servers))
    last = (a**servers) / (math.factorial(servers) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last)
    lq = p0 * (a**servers) * rho / (math.factorial(servers) * (1.0 - rho) ** 2)
    wq = lq / arrival
    w = wq + 1.0 / service
    l = arrival * w
    return {"utilization": rho, "p0": p0, "lq": lq, "wq": wq, "l": l, "w": w, "stable": 1.0}


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None


def _positive_int(value: float) -> int:
    if value <= 0 or pd.isna(value):
        return 1
    return max(1, int(round(value)))


def _to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
