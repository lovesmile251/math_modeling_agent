from __future__ import annotations

import pandas as pd


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[1] < 2:
        return pd.DataFrame()

    target_column = _choose_target(numeric)
    feature_columns = [column for column in numeric.columns if column != target_column]
    if not feature_columns:
        return pd.DataFrame()

    variances = numeric[feature_columns].var(ddof=0)
    max_variance = float(variances.max()) if len(variances) else 0.0
    corr = numeric[feature_columns + [target_column]].corr(method="pearson")[target_column].drop(target_column)

    rows: list[dict[str, float | str | int]] = []
    for column in feature_columns:
        variance_score = float(variances[column] / max_variance) if max_variance > 0 else 0.0
        correlation = float(corr.get(column, 0.0))
        score = 0.6 * abs(correlation) + 0.4 * variance_score
        rows.append(
            {
                "target": str(target_column),
                "feature": str(column),
                "variance": float(variances[column]),
                "variance_score": variance_score,
                "target_correlation": correlation,
                "importance_score": float(score),
                "rank": 0,
            }
        )

    result = pd.DataFrame(rows).sort_values("importance_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def _choose_target(df: pd.DataFrame) -> str:
    priority_keywords = ("target", "y", "demand", "sales", "profit", "revenue", "score")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in priority_keywords):
            return str(column)
    return str(df.columns[-1])
