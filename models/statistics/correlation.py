from __future__ import annotations

import pandas as pd


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return pd.DataFrame()

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    rows: list[dict[str, float | str]] = []
    columns = list(numeric.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            rows.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "pearson": float(pearson.loc[left, right]),
                    "spearman": float(spearman.loc[left, right]),
                    "abs_pearson": abs(float(pearson.loc[left, right])),
                    "abs_spearman": abs(float(spearman.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_pearson", ascending=False).reset_index(drop=True)
