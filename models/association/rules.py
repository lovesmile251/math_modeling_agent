from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def apriori_rules(df: pd.DataFrame, min_support: float = 0.2, min_confidence: float = 0.6) -> pd.DataFrame:
    transactions = _transactions(df)
    if len(transactions) < 2:
        return pd.DataFrame()

    item_counts: dict[frozenset[str], int] = {}
    max_size = min(3, max(len(transaction) for transaction in transactions))
    for size in range(1, max_size + 1):
        candidates = set()
        for transaction in transactions:
            for combo in itertools.combinations(sorted(transaction), size):
                candidates.add(frozenset(combo))
        for candidate in candidates:
            count = sum(1 for transaction in transactions if candidate.issubset(transaction))
            support = count / len(transactions)
            if support >= min_support:
                item_counts[candidate] = count

    rows: list[dict[str, float | str | int]] = []
    for itemset, count in item_counts.items():
        if len(itemset) < 2:
            continue
        support = count / len(transactions)
        for antecedent_size in range(1, len(itemset)):
            for antecedent_tuple in itertools.combinations(sorted(itemset), antecedent_size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_count = item_counts.get(antecedent) or sum(1 for t in transactions if antecedent.issubset(t))
                consequent_count = item_counts.get(consequent) or sum(1 for t in transactions if consequent.issubset(t))
                if antecedent_count == 0 or consequent_count == 0:
                    continue
                confidence = count / antecedent_count
                if confidence < min_confidence:
                    continue
                consequent_support = consequent_count / len(transactions)
                rows.append(
                    {
                        "antecedent": ", ".join(sorted(antecedent)),
                        "consequent": ", ".join(sorted(consequent)),
                        "support": support,
                        "confidence": confidence,
                        "lift": confidence / consequent_support if consequent_support > 0 else 0.0,
                        "transaction_count": len(transactions),
                        "method": "apriori",
                    }
                )
    return pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def granger_causality(df: pd.DataFrame, max_lag: int = 2) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < max_lag + 5 or numeric.shape[1] < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    columns = list(numeric.columns)
    for cause in columns:
        for target in columns:
            if cause == target:
                continue
            best = _granger_pair(numeric[cause].to_numpy(float), numeric[target].to_numpy(float), max_lag)
            if best is not None:
                rows.append(
                    {
                        "cause": str(cause),
                        "target": str(target),
                        **best,
                        "method": "lagged_regression_granger_approx",
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("f_statistic", ascending=False).reset_index(drop=True)


def _transactions(df: pd.DataFrame) -> list[set[str]]:
    transactions: list[set[str]] = []
    for _, row in df.iterrows():
        items: set[str] = set()
        for column, value in row.items():
            if pd.isna(value):
                continue
            if isinstance(value, str):
                parts = [part.strip() for part in value.replace(";", ",").replace("|", ",").split(",")]
                if len(parts) > 1:
                    items.update(part for part in parts if part)
                elif value.strip():
                    items.add(f"{column}={value.strip()}")
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    items.add(str(column))
        if items:
            transactions.append(items)
    return transactions


def _granger_pair(cause: np.ndarray, target: np.ndarray, max_lag: int) -> dict[str, float | int] | None:
    best: dict[str, float | int] | None = None
    for lag in range(1, max_lag + 1):
        y = target[lag:]
        target_lags = np.column_stack([target[lag - i - 1 : len(target) - i - 1] for i in range(lag)])
        cause_lags = np.column_stack([cause[lag - i - 1 : len(cause) - i - 1] for i in range(lag)])
        restricted = np.column_stack([np.ones(len(y)), target_lags])
        unrestricted = np.column_stack([restricted, cause_lags])
        rss_restricted = _rss(restricted, y)
        rss_unrestricted = _rss(unrestricted, y)
        df_num = lag
        df_den = len(y) - unrestricted.shape[1]
        if df_den <= 0 or rss_unrestricted <= 0:
            continue
        f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_den)
        improvement = 0.0 if rss_restricted <= 0 else max(0.0, 1 - rss_unrestricted / rss_restricted)
        item = {
            "lag": lag,
            "f_statistic": float(max(f_stat, 0.0)),
            "rss_restricted": float(rss_restricted),
            "rss_unrestricted": float(rss_unrestricted),
            "relative_rss_improvement": float(improvement),
            "sample_size": int(len(y)),
        }
        if best is None or float(item["f_statistic"]) > float(best["f_statistic"]):
            best = item
    return best


def _rss(x: np.ndarray, y: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    return float(np.sum(residuals**2))
