from __future__ import annotations

import numpy as np
import pandas as pd


def ahp_entropy_combined_weights(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.empty:
        return pd.DataFrame()

    ahp = _variance_weights(numeric)
    entropy = _entropy_weights(numeric)
    combined = ahp * entropy
    if float(combined.sum()) == 0.0:
        combined = pd.Series(1.0 / len(numeric.columns), index=numeric.columns, dtype=float)
    else:
        combined = combined / combined.sum()

    return pd.DataFrame(
        {
            "indicator": [str(column) for column in numeric.columns],
            "ahp_weight": ahp.reindex(numeric.columns).to_numpy(dtype=float),
            "entropy_weight": entropy.reindex(numeric.columns).to_numpy(dtype=float),
            "combined_weight": combined.reindex(numeric.columns).to_numpy(dtype=float),
            "method": "AHP_entropy_combined_weights",
        }
    )


def dea_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] == 0 or numeric.shape[1] < 2:
        return pd.DataFrame()

    input_columns = _columns_by_keywords(
        numeric,
        ("input", "cost", "resource", "capital", "labor", "staff", "employee", "expense"),
    )
    output_columns = _columns_by_keywords(
        numeric,
        ("output", "revenue", "profit", "sales", "income", "throughput", "score", "yield"),
    )
    if not input_columns or not output_columns:
        columns = list(numeric.columns)
        midpoint = max(1, len(columns) // 2)
        input_columns = columns[:midpoint]
        output_columns = columns[midpoint:]
    if not input_columns or not output_columns:
        return pd.DataFrame()

    inputs = numeric[input_columns].clip(lower=0.0)
    outputs = numeric[output_columns].clip(lower=0.0)
    input_score = _weighted_mean_score(inputs, inverse=True)
    output_score = _weighted_mean_score(outputs, inverse=False)
    raw_efficiency = np.divide(
        output_score,
        input_score,
        out=np.zeros_like(output_score, dtype=float),
        where=input_score > 0,
    )
    max_efficiency = float(np.nanmax(raw_efficiency)) if len(raw_efficiency) else 0.0
    efficiency = raw_efficiency if max_efficiency <= 0 else raw_efficiency / max_efficiency

    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "input_score": input_score,
            "output_score": output_score,
            "efficiency": efficiency,
            "efficient": (efficiency >= 1.0 - 1e-9).astype(float),
            "input_columns": ", ".join(str(column) for column in input_columns),
            "output_columns": ", ".join(str(column) for column in output_columns),
            "method": "DEA_CCR_ratio_envelopment",
        }
    )


def fuzzy_comprehensive_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.empty:
        return pd.DataFrame()

    normalized = _minmax_normalize(numeric)
    weights = _entropy_weights(numeric).reindex(numeric.columns).fillna(0.0)
    if float(weights.sum()) == 0.0:
        weights = pd.Series(1.0 / len(numeric.columns), index=numeric.columns, dtype=float)
    else:
        weights = weights / weights.sum()

    low = (1.0 - normalized).clip(lower=0.0, upper=1.0)
    medium = (1.0 - (normalized - 0.5).abs() * 2.0).clip(lower=0.0, upper=1.0)
    high = normalized.clip(lower=0.0, upper=1.0)
    membership_sum = low + medium + high
    membership_sum = membership_sum.replace(0.0, 1.0)
    low = low / membership_sum
    medium = medium / membership_sum
    high = high / membership_sum

    w = weights.to_numpy(dtype=float)
    low_score = low.to_numpy(dtype=float) @ w
    medium_score = medium.to_numpy(dtype=float) @ w
    high_score = high.to_numpy(dtype=float) @ w
    fuzzy_score = medium_score * 0.5 + high_score
    labels = np.array(["low", "medium", "high"], dtype=object)
    grade_index = np.column_stack([low_score, medium_score, high_score]).argmax(axis=1)

    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "membership_low": low_score,
            "membership_medium": medium_score,
            "membership_high": high_score,
            "fuzzy_score": fuzzy_score,
            "grade": labels[grade_index],
            "method": "fuzzy_comprehensive_evaluation",
        }
    )


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.apply(pd.to_numeric, errors="coerce").fillna(numeric.mean(numeric_only=True)).fillna(0.0)


def _variance_weights(df: pd.DataFrame) -> pd.Series:
    std = df.std(numeric_only=True).fillna(0.0).astype(float)
    if float(std.sum()) == 0.0:
        return pd.Series(1.0 / len(std), index=std.index, dtype=float)
    return std / std.sum()


def _entropy_weights(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    normalized = _minmax_normalize(df)
    if len(normalized) <= 1:
        return pd.Series(1.0 / len(normalized.columns), index=normalized.columns, dtype=float)

    matrix = normalized.to_numpy(dtype=float) + 1e-12
    column_sums = matrix.sum(axis=0)
    probabilities = matrix / column_sums
    entropy = -np.sum(probabilities * np.log(probabilities), axis=0) / np.log(len(normalized))
    diversity = 1.0 - entropy
    if np.allclose(diversity.sum(), 0.0):
        weights = np.ones_like(diversity) / len(diversity)
    else:
        weights = diversity / diversity.sum()
    return pd.Series(weights, index=normalized.columns, dtype=float)


def _minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)
    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce").astype(float)
        min_value = series.min()
        max_value = series.max()
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            normalized[column] = 0.0
        else:
            normalized[column] = (series - min_value) / (max_value - min_value)
    return normalized.fillna(0.0)


def _columns_by_keywords(df: pd.DataFrame, keywords: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            columns.append(str(column))
    return columns


def _weighted_mean_score(df: pd.DataFrame, inverse: bool) -> np.ndarray:
    normalized = _minmax_normalize(df)
    if inverse:
        normalized = 1.0 - normalized
    weights = _entropy_weights(df).reindex(df.columns).fillna(0.0)
    if float(weights.sum()) == 0.0:
        weights = pd.Series(1.0 / len(df.columns), index=df.columns, dtype=float)
    else:
        weights = weights / weights.sum()
    return normalized.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
