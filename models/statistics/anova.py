from __future__ import annotations

import pandas as pd


def anova_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = list(df.select_dtypes(include="number").columns)
    category_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and 2 <= df[column].dropna().nunique() <= 20
    ]
    if not numeric_columns or not category_columns:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for category in category_columns[:5]:
        grouped = df[[category, *numeric_columns]].dropna(subset=[category])
        for value_column in numeric_columns:
            groups = [
                group[value_column].dropna().astype(float)
                for _, group in grouped.groupby(category, dropna=True)
                if len(group[value_column].dropna()) > 0
            ]
            if len(groups) < 2 or sum(len(group) for group in groups) <= len(groups):
                continue

            all_values = pd.concat(groups)
            grand_mean = float(all_values.mean())
            ss_between = sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in groups)
            ss_within = sum(float(((group - float(group.mean())) ** 2).sum()) for group in groups)
            df_between = len(groups) - 1
            df_within = len(all_values) - len(groups)
            ms_between = ss_between / df_between if df_between else 0.0
            ms_within = ss_within / df_within if df_within else 0.0
            f_stat = ms_between / ms_within if ms_within > 0 else float("inf")

            rows.append(
                {
                    "factor": str(category),
                    "response": str(value_column),
                    "groups": int(len(groups)),
                    "sample_size": int(len(all_values)),
                    "df_between": int(df_between),
                    "df_within": int(df_within),
                    "ss_between": float(ss_between),
                    "ss_within": float(ss_within),
                    "f_statistic": float(f_stat),
                }
            )
    return pd.DataFrame(rows).sort_values("f_statistic", ascending=False).reset_index(drop=True)
