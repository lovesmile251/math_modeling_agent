from __future__ import annotations

import numpy as np
import pandas as pd


def crop_planting_plan(df: pd.DataFrame, land_use_ratio: float = 0.85) -> pd.DataFrame:
    """Greedy crop planting plan with interpretable area and profit outputs."""

    if df.empty:
        return pd.DataFrame()

    crop_col = _find_text_column(df, ("crop", "plant", "name", "作物", "品种", "农作物"))
    land_col = _find_text_column(df, ("land", "plot", "field", "地块", "土地", "田块"))
    season_col = _find_text_column(df, ("season", "quarter", "month", "季节", "季度", "茬"))
    area_col = _find_numeric_column(df, ("area", "acre", "mu", "land", "面积", "亩"))
    yield_col = _find_numeric_column(df, ("yield", "output", "production", "产量", "亩产"))
    price_col = _find_numeric_column(df, ("price", "revenue", "sale", "售价", "价格", "收入"))
    cost_col = _find_numeric_column(df, ("cost", "expense", "成本", "费用"))
    demand_col = _find_numeric_column(df, ("demand", "market", "sales", "需求", "销量"))
    profit_col = _find_numeric_column(df, ("profit", "margin", "benefit", "利润", "收益"))

    work = pd.DataFrame({"row_index": df.index})
    work["crop"] = df[crop_col].astype(str) if crop_col else df.index.map(lambda idx: f"crop_{idx}")
    work["land"] = df[land_col].astype(str) if land_col else ""
    work["season"] = df[season_col].astype(str) if season_col else ""
    work["max_area"] = _numeric_or_default(df, area_col, 1.0)
    work["yield_per_area"] = _numeric_or_default(df, yield_col, 1.0)
    work["price"] = _numeric_or_default(df, price_col, 1.0)
    work["cost_per_area"] = _numeric_or_default(df, cost_col, 0.0)
    work["demand"] = _numeric_or_default(df, demand_col, np.inf)
    if profit_col:
        work["profit_per_area"] = pd.to_numeric(df[profit_col], errors="coerce")
    else:
        work["profit_per_area"] = work["yield_per_area"] * work["price"] - work["cost_per_area"]

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["max_area", "yield_per_area", "profit_per_area"])
    work = work[(work["max_area"] > 0) & (work["yield_per_area"] > 0)]
    if work.empty:
        return pd.DataFrame()

    work["demand_limited_area"] = np.divide(
        work["demand"],
        work["yield_per_area"],
        out=np.full(len(work), np.inf),
        where=work["yield_per_area"].to_numpy(float) > 0,
    )
    work["candidate_area"] = np.minimum(work["max_area"], work["demand_limited_area"])
    work["candidate_area"] = np.where(np.isfinite(work["candidate_area"]), work["candidate_area"], work["max_area"])
    work = work[work["candidate_area"] > 0].sort_values("profit_per_area", ascending=False).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    total_land = float(work["max_area"].sum() * land_use_ratio)
    remaining = max(total_land, float(work["candidate_area"].min()))
    allocated = []
    for area in work["candidate_area"]:
        value = float(min(area, max(remaining, 0.0)))
        allocated.append(value)
        remaining -= value

    work["priority_rank"] = np.arange(1, len(work) + 1)
    work["allocated_area"] = allocated
    work["expected_production"] = work["allocated_area"] * work["yield_per_area"]
    work["expected_revenue"] = work["expected_production"] * work["price"]
    work["expected_cost"] = work["allocated_area"] * work["cost_per_area"]
    work["expected_profit"] = work["allocated_area"] * work["profit_per_area"]
    work["demand_satisfaction_rate"] = np.divide(
        work["expected_production"],
        work["demand"],
        out=np.full(len(work), np.nan),
        where=np.isfinite(work["demand"].to_numpy(float)) & (work["demand"].to_numpy(float) > 0),
    )
    work["total_available_land"] = total_land
    work["land_use_ratio"] = land_use_ratio
    work["method"] = "profit_density_greedy_crop_planting"
    return work[
        [
            "priority_rank",
            "row_index",
            "crop",
            "land",
            "season",
            "max_area",
            "allocated_area",
            "yield_per_area",
            "expected_production",
            "price",
            "expected_revenue",
            "expected_cost",
            "expected_profit",
            "demand_satisfaction_rate",
            "total_available_land",
            "method",
        ]
    ]


def _find_text_column(df: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    non_numeric = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    return _find_column(non_numeric, terms)


def _find_numeric_column(df: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    numeric = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    return _find_column(numeric, terms)


def _find_column(columns: list[object], terms: tuple[str, ...]) -> str | None:
    for column in columns:
        name = str(column).lower()
        if any(term.lower() in name for term in terms):
            return str(column)
    return None


def _numeric_or_default(df: pd.DataFrame, column: str | None, default: float) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = pd.to_numeric(df[column], errors="coerce")
    return values.fillna(default)
