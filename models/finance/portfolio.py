from __future__ import annotations

import math

import numpy as np
import pandas as pd


def black_scholes_pricing(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()

    spot_col = _find_column(numeric, ("spot", "underlying", "price", "s0", "标的", "现价", "价格"))
    strike_col = _find_column(numeric, ("strike", "exercise", "k", "行权", "执行"))
    rate_col = _find_column(numeric, ("rate", "risk_free", "r", "利率", "无风险"))
    vol_col = _find_column(numeric, ("vol", "sigma", "波动"))
    time_col = _find_column(numeric, ("time", "maturity", "tau", "期限", "到期"))
    if spot_col is None or strike_col is None:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for idx, row in df.iterrows():
        spot = _to_float(row.get(spot_col))
        strike = _to_float(row.get(strike_col))
        rate = _to_float(row.get(rate_col), 0.03) if rate_col else 0.03
        volatility = _to_float(row.get(vol_col), 0.2) if vol_col else 0.2
        time_to_maturity = _to_float(row.get(time_col), 1.0) if time_col else 1.0
        if spot <= 0 or strike <= 0 or volatility <= 0 or time_to_maturity <= 0:
            continue
        sqrt_t = math.sqrt(time_to_maturity)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        call = spot * _normal_cdf(d1) - strike * math.exp(-rate * time_to_maturity) * _normal_cdf(d2)
        put = strike * math.exp(-rate * time_to_maturity) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
        rows.append(
            {
                "row_index": idx,
                "spot": spot,
                "strike": strike,
                "risk_free_rate": rate,
                "volatility": volatility,
                "time_to_maturity": time_to_maturity,
                "d1": d1,
                "d2": d2,
                "call_price": call,
                "put_price": put,
                "method": "black_scholes",
            }
        )
    return pd.DataFrame(rows)


def markowitz_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    returns = _returns_frame(df)
    if returns.shape[0] < 3 or returns.shape[1] < 2:
        return pd.DataFrame()

    expected = returns.mean().to_numpy(dtype=float)
    covariance = returns.cov().to_numpy(dtype=float)
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    inv_cov = np.linalg.pinv(covariance)
    ones = np.ones(len(expected))

    min_var_raw = inv_cov @ ones
    min_var_weights = _long_only_normalize(min_var_raw)
    tangency_raw = inv_cov @ expected
    tangency_weights = _long_only_normalize(tangency_raw)
    if tangency_weights.sum() == 0:
        tangency_weights = min_var_weights

    rows = []
    for portfolio_name, weights in (("min_variance", min_var_weights), ("return_risk", tangency_weights)):
        port_return = float(weights @ expected)
        port_vol = float(math.sqrt(max(weights @ covariance @ weights, 0.0)))
        sharpe_like = port_return / port_vol if port_vol > 0 else 0.0
        for asset, weight, asset_return in zip(returns.columns, weights, expected):
            rows.append(
                {
                    "portfolio": portfolio_name,
                    "asset": str(asset),
                    "weight": float(weight),
                    "expected_return": float(asset_return),
                    "portfolio_expected_return": port_return,
                    "portfolio_volatility": port_vol,
                    "return_risk_ratio": sharpe_like,
                    "method": "markowitz_long_only_pinv",
                }
            )
    return pd.DataFrame(rows)


def _returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return pd.DataFrame()

    returns = pd.DataFrame(index=numeric.index)
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce")
        name = str(column).lower()
        if "return" in name or "收益率" in name or "ret" == name:
            candidate = series
        elif series.dropna().abs().median() > 2:
            candidate = series.pct_change(fill_method=None)
        else:
            candidate = series
        candidate = candidate.replace([np.inf, -np.inf], np.nan)
        if candidate.dropna().std() > 0:
            returns[str(column)] = candidate
    return returns.dropna()


def _long_only_normalize(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return np.ones(len(values)) / len(values)
    return clipped / total


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword == name or keyword in name for keyword in keywords):
            return str(column)
    return None


def _to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
