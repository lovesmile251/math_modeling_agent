from __future__ import annotations

import math

import pandas as pd


def inventory_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Return an EOQ/replenishment policy table for inventory-style data.

    The model supports both textbook inventory fields and CUMCM-style retail
    data. It can use explicit demand/order/holding costs when available; when
    only partial signals such as sales volume, wholesale price, loss rate, item
    code, or category are present, it still returns a conservative proxy policy
    instead of failing silently.
    """

    demand_col = _find_column(
        df,
        ("demand", "需求", "销量", "销售量", "销售", "sell", "sales", "quantity", "qty"),
    )
    order_col = _find_column(
        df,
        ("order_cost", "ordering_cost", "order", "setup", "订货", "订购", "补货", "启动"),
    )
    holding_col = _find_column(
        df,
        ("holding", "holding_cost", "storage", "库存持有", "持有", "仓储", "保管"),
    )
    unit_cost_col = _find_column(
        df,
        ("unit_cost", "wholesale", "price", "cost", "批发价格", "单价", "价格", "成本"),
    )
    loss_col = _find_column(df, ("loss", "waste", "损耗", "损耗率", "损失率", "报损"))
    stock_col = _find_column(df, ("inventory", "stock", "库存", "存量"))
    group_col = _find_column(
        df,
        ("item", "sku", "product", "category", "品类", "分类", "单品", "编码", "名称", "小分类"),
    )

    if demand_col is None and unit_cost_col is None and loss_col is None and stock_col is None:
        return pd.DataFrame()

    groups = df.groupby(group_col, dropna=False) if group_col else [(None, df)]
    rows: list[dict[str, float | int | str]] = []
    for key, group in groups:
        demand_series = _numeric_series(group[demand_col]) if demand_col else pd.Series(dtype=float)
        unit_cost_series = (
            _numeric_series(group[unit_cost_col]) if unit_cost_col else pd.Series(dtype=float)
        )
        loss_series = _numeric_series(group[loss_col]) if loss_col else pd.Series(dtype=float)
        stock_series = _numeric_series(group[stock_col]) if stock_col else pd.Series(dtype=float)

        demand = _positive_mean(
            demand_series,
            default=float(len(group)) if demand_col is None else 0.0,
        )
        demand_std = _positive_std(demand_series)
        unit_cost = _positive_mean(unit_cost_series, default=1.0)
        loss_rate = _positive_mean(loss_series, default=0.0)
        current_stock = _positive_mean(stock_series, default=0.0)

        order_cost = _positive_mean(
            _numeric_series(group[order_col]) if order_col else pd.Series(dtype=float),
            default=max(unit_cost * 2.0, 1.0),
        )
        holding_cost = _positive_mean(
            _numeric_series(group[holding_col]) if holding_col else pd.Series(dtype=float),
            default=max(unit_cost * max(loss_rate / 100.0, 0.08), 0.01),
        )

        if demand <= 0 or order_cost <= 0 or holding_cost <= 0:
            continue

        eoq = math.sqrt(2 * demand * order_cost / holding_cost)
        safety_stock = 1.65 * demand_std if demand_std > 0 else max(0.1 * demand, 1.0)
        reorder_point = demand + safety_stock
        shrinkage_multiplier = 1.0 / max(1.0 - min(max(loss_rate, 0.0), 95.0) / 100.0, 0.05)
        suggested_replenishment = max(eoq, reorder_point - current_stock) * shrinkage_multiplier
        ordering_cost = demand / eoq * order_cost
        inventory_cost = eoq / 2 * holding_cost

        rows.append(
            {
                "item_key": "" if key is None else str(key),
                "records": int(len(group)),
                "demand_proxy": demand,
                "demand_std": demand_std,
                "order_cost": order_cost,
                "holding_cost": holding_cost,
                "mean_unit_cost": unit_cost,
                "loss_rate_percent": loss_rate,
                "current_stock_proxy": current_stock,
                "eoq": eoq,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "suggested_replenishment": suggested_replenishment,
                "price_markup_floor": shrinkage_multiplier,
                "estimated_total_cost": ordering_cost + inventory_cost + demand * unit_cost,
                "method": "EOQ_or_replenishment_proxy",
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["suggested_replenishment", "estimated_total_cost"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword.lower() in name for keyword in keywords):
            return str(column)
    return None


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _positive_mean(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    positive = series[series > 0]
    if positive.empty:
        return default
    return float(positive.mean())


def _positive_std(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    value = float(series.std(ddof=0))
    return value if math.isfinite(value) and value > 0 else 0.0
