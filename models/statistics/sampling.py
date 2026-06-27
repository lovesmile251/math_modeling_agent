from __future__ import annotations

import math

import pandas as pd


def quality_sampling_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Build binomial sampling plans for defect-rate inspection problems."""

    target_rate = _target_defect_rate(df)
    confidence_levels = (0.90, 0.95)
    relative_margins = (0.20, 0.10)
    rows: list[dict[str, float | int | str]] = []
    for confidence in confidence_levels:
        z = _z_value(confidence)
        for relative_margin in relative_margins:
            margin = max(target_rate * relative_margin, 0.005)
            n = math.ceil(z * z * target_rate * (1.0 - target_rate) / (margin * margin))
            accept_threshold = max(0, math.floor(n * target_rate - z * math.sqrt(n * target_rate * (1 - target_rate))))
            reject_threshold = math.ceil(n * target_rate + z * math.sqrt(n * target_rate * (1 - target_rate)))
            rows.append(
                {
                    "target_defect_rate": round(target_rate, 6),
                    "confidence": confidence,
                    "relative_margin": relative_margin,
                    "absolute_margin": round(margin, 6),
                    "sample_size": int(n),
                    "accept_if_defects_leq": int(accept_threshold),
                    "reject_if_defects_geq": int(reject_threshold),
                    "accept_defect_rate_threshold": round(accept_threshold / n, 6),
                    "reject_defect_rate_threshold": round(reject_threshold / n, 6),
                    "method": "normal_approx_binomial_quality_sampling",
                }
            )
    return pd.DataFrame(rows)


def _target_defect_rate(df: pd.DataFrame) -> float:
    lower_columns = {str(column).lower(): column for column in df.columns}
    for token in ("defect_rate", "nonconforming_rate", "bad_rate", "次品率", "不合格率"):
        column = lower_columns.get(token)
        if column is None:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if not series.empty:
            value = float(series.mean())
            return _clip_rate(value)

    defect_col = _find_column(df, ("defect", "bad", "nonconforming", "次品", "不合格"))
    sample_col = _find_column(df, ("sample", "total", "count", "样本", "总数", "数量"))
    if defect_col and sample_col:
        defects = pd.to_numeric(df[defect_col], errors="coerce").sum()
        total = pd.to_numeric(df[sample_col], errors="coerce").sum()
        if total > 0:
            return _clip_rate(float(defects / total))

    return 0.10


def _find_column(df: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    for column in df.columns:
        lower = str(column).lower()
        if any(term.lower() in lower for term in terms):
            return str(column)
    return None


def _clip_rate(value: float) -> float:
    if value > 1.0:
        value = value / 100.0
    return min(max(value, 0.001), 0.499)


def _z_value(confidence: float) -> float:
    if confidence >= 0.99:
        return 2.576
    if confidence >= 0.95:
        return 1.96
    return 1.645
