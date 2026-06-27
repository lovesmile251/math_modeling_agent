from __future__ import annotations

import numpy as np
import pandas as pd


def var_cvar_risk(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    returns = _returns_frame(df)
    if returns.empty:
        return pd.DataFrame()

    alpha = 1.0 - confidence
    rows: list[dict[str, float | str | int]] = []
    for column in returns.columns:
        series = pd.to_numeric(returns[column], errors="coerce").dropna()
        if len(series) < 3:
            continue
        values = series.to_numpy(dtype=float)
        threshold = float(np.quantile(values, alpha))
        tail = values[values <= threshold]
        var_value = -threshold
        cvar_value = -float(tail.mean()) if len(tail) else var_value
        rows.append(
            {
                "asset": str(column),
                "sample_size": int(len(values)),
                "confidence": confidence,
                "mean_return": float(values.mean()),
                "volatility": float(values.std(ddof=1)),
                "var": float(var_value),
                "cvar": float(cvar_value),
                "worst_return": float(values.min()),
                "method": "historical_var_cvar",
            }
        )
    return pd.DataFrame(rows).sort_values("cvar", ascending=False).reset_index(drop=True)


def garch_volatility(df: pd.DataFrame, alpha: float = 0.1, beta: float = 0.85) -> pd.DataFrame:
    returns = _returns_frame(df)
    if returns.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in returns.columns:
        series = pd.to_numeric(returns[column], errors="coerce").dropna()
        if len(series) < 5:
            continue
        values = series.to_numpy(dtype=float)
        variance = float(np.var(values, ddof=1))
        omega = max(variance * (1.0 - alpha - beta), 1e-12)
        sigma2 = max(variance, 1e-12)
        for value in values:
            sigma2 = omega + alpha * float(value**2) + beta * sigma2
        rows.append(
            {
                "asset": str(column),
                "sample_size": int(len(values)),
                "mean_return": float(values.mean()),
                "latest_conditional_volatility": float(np.sqrt(sigma2)),
                "annualized_volatility_252": float(np.sqrt(sigma2) * np.sqrt(252)),
                "alpha": alpha,
                "beta": beta,
                "omega": omega,
                "method": "garch_1_1_approximation",
            }
        )
    return pd.DataFrame(rows).sort_values("latest_conditional_volatility", ascending=False).reset_index(drop=True)


def _returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] == 0:
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
    return returns.dropna(axis=1, how="all")
