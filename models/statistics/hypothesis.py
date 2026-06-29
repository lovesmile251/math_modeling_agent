from __future__ import annotations

import math

import numpy as np
import pandas as pd


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    rows: list[dict[str, float | str | int]] = []

    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < 2:
            continue
        values = series.to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if std <= 0:
            continue
        t_stat = mean / (std / math.sqrt(len(values)))
        rows.append(
            {
                "test": "one_sample_t_zero",
                "variable": str(column),
                "sample_size": int(len(values)),
                "statistic": float(t_stat),
                "normal_approx_p_value": _two_sided_normal_p(t_stat),
                "null_hypothesis": "mean = 0",
            }
        )

    categorical = df.select_dtypes(exclude="number").dropna(axis=1, how="all")
    for column in categorical.columns:
        counts = df[column].dropna().astype(str).value_counts()
        if len(counts) < 2:
            continue
        expected = counts.sum() / len(counts)
        chi_square = float(((counts - expected) ** 2 / expected).sum()) if expected > 0 else 0.0
        rows.append(
            {
                "test": "chi_square_uniform",
                "variable": str(column),
                "sample_size": int(counts.sum()),
                "statistic": chi_square,
                "normal_approx_p_value": float("nan"),
                "null_hypothesis": "category frequencies are uniform",
            }
        )

    return pd.DataFrame(rows)


def statistical_test_suite(df: pd.DataFrame) -> pd.DataFrame:
    """Run a compact suite of inferential checks with effect-size fields."""

    rows: list[dict[str, float | str | int]] = []
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric_cols = [str(column) for column in numeric.columns]

    for column in numeric_cols:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < 3:
            continue
        values = series.to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if std <= 0:
            continue
        skew = float(((values - mean) ** 3).mean() / (std ** 3))
        kurtosis = float(((values - mean) ** 4).mean() / (std ** 4))
        jb_stat = len(values) / 6.0 * (skew * skew + 0.25 * (kurtosis - 3.0) ** 2)
        t_stat = mean / (std / math.sqrt(len(values)))
        rows.append(
            {
                "test": "one_sample_t_zero",
                "variable": column,
                "sample_size": int(len(values)),
                "statistic": float(t_stat),
                "p_value": _two_sided_normal_p(t_stat),
                "effect_size": float(mean / std),
                "null_hypothesis": "mean = 0",
                "multiple_testing_note": "apply BH/FDR if many variables are interpreted",
            }
        )
        rows.append(
            {
                "test": "jarque_bera_normality_approx",
                "variable": column,
                "sample_size": int(len(values)),
                "statistic": float(jb_stat),
                "p_value": math.exp(-0.5 * max(jb_stat, 0.0)),
                "effect_size": abs(skew) + abs(kurtosis - 3.0),
                "null_hypothesis": "normal distribution",
                "multiple_testing_note": "screening diagnostic, not a proof of normality",
            }
        )

    if len(numeric_cols) >= 2:
        for left, right in zip(numeric_cols, numeric_cols[1:]):
            pair = numeric[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 4:
                continue
            corr = float(pair[left].corr(pair[right]))
            if not math.isfinite(corr):
                continue
            denom = max(1e-12, 1.0 - corr * corr)
            t_stat = corr * math.sqrt((len(pair) - 2) / denom)
            rows.append(
                {
                    "test": "pearson_correlation_significance",
                    "variable": f"{left}~{right}",
                    "sample_size": int(len(pair)),
                    "statistic": float(t_stat),
                    "p_value": _two_sided_normal_p(t_stat),
                    "effect_size": abs(corr),
                    "null_hypothesis": "correlation = 0",
                    "multiple_testing_note": "adjust p-values when scanning many pairs",
                }
            )

    categorical = df.select_dtypes(exclude="number").dropna(axis=1, how="all")
    if not categorical.empty and numeric_cols:
        group_col = str(categorical.columns[0])
        target_col = numeric_cols[0]
        groups = []
        for _name, group in df[[group_col, target_col]].dropna().groupby(group_col):
            values = pd.to_numeric(group[target_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) >= 2:
                groups.append(values)
        if len(groups) >= 2:
            left, right = groups[0], groups[1]
            pooled = math.sqrt(
                max(
                    1e-12,
                    ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
                    / max(len(left) + len(right) - 2, 1),
                )
            )
            diff = float(left.mean() - right.mean())
            se = pooled * math.sqrt(1 / len(left) + 1 / len(right))
            t_stat = diff / se if se > 0 else 0.0
            rows.append(
                {
                    "test": "two_sample_t_approx",
                    "variable": f"{target_col} by {group_col}",
                    "sample_size": int(len(left) + len(right)),
                    "statistic": float(t_stat),
                    "p_value": _two_sided_normal_p(t_stat),
                    "effect_size": float(diff / pooled) if pooled > 0 else 0.0,
                    "null_hypothesis": "group means are equal",
                    "multiple_testing_note": "check group balance and variance assumptions",
                }
            )

    return pd.DataFrame(rows)


def _two_sided_normal_p(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))
