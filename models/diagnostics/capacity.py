from __future__ import annotations

import pandas as pd


def demand_capacity_gap(df: pd.DataFrame) -> pd.DataFrame:
    demand = _find_column(df, ("demand", "need", "sales", "volume"))
    capacity = _find_column(df, ("capacity", "supply", "capability", "limit"))
    if demand is None or capacity is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=df.index)
    result["row_index"] = df.index
    result["demand_column"] = demand
    result["capacity_column"] = capacity
    result["demand"] = pd.to_numeric(df[demand], errors="coerce")
    result["capacity"] = pd.to_numeric(df[capacity], errors="coerce")
    result["gap"] = result["capacity"] - result["demand"]
    result["utilization"] = result["demand"] / result["capacity"].replace(0, pd.NA)
    result["status"] = result["gap"].apply(lambda value: "shortage" if pd.notna(value) and value < 0 else "sufficient")
    return result


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None
