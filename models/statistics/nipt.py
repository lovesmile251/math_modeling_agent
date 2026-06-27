from __future__ import annotations

import numpy as np
import pandas as pd


def nipt_bmi_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """Build BMI groups and recommended NIPT test weeks.

    The model is intentionally interpretable: it groups pregnant subjects by BMI
    and estimates the earliest gestational week where Y chromosome concentration
    reaches the common 4% reliability threshold. If Y concentration or week data
    is absent, it falls back to BMI-based risk timing.
    """

    if df.empty:
        return pd.DataFrame()

    bmi_column = _find_column(df, ("bmi", "body_mass_index", "体重指数", "孕妇bmi"))
    if bmi_column is None:
        return pd.DataFrame()

    week_column = _find_column(df, ("week", "gestational", "ga", "孕周", "检测孕周"))
    y_column = _find_column(df, ("y_concentration", "y_chromosome", "ychromosome", "y染色体", "y"))
    abnormal_column = _find_column(df, ("abnormal", "risk", "aneuploidy", "胎儿异常", "异常"))

    work = pd.DataFrame({"bmi": pd.to_numeric(df[bmi_column], errors="coerce")})
    if week_column is not None:
        work["week"] = pd.to_numeric(df[week_column], errors="coerce")
    if y_column is not None:
        work["y_concentration"] = pd.to_numeric(df[y_column], errors="coerce")
    if abnormal_column is not None:
        work["abnormal"] = pd.to_numeric(df[abnormal_column], errors="coerce")
    work = work.dropna(subset=["bmi"])
    if len(work) < 3:
        return pd.DataFrame()

    work["bmi_group"] = _bmi_groups(work["bmi"])
    rows: list[dict[str, float | int | str]] = []
    for group, part in work.groupby("bmi_group", observed=False):
        bmi_mean = float(part["bmi"].mean())
        recommended_week, reach_rate = _recommended_week(part, bmi_mean)
        rows.append(
            {
                "bmi_group": str(group),
                "sample_size": int(len(part)),
                "bmi_min": float(part["bmi"].min()),
                "bmi_max": float(part["bmi"].max()),
                "bmi_mean": bmi_mean,
                "recommended_week": recommended_week,
                "y_threshold": 0.04,
                "threshold_reach_rate": reach_rate,
                "abnormal_rate": _abnormal_rate(part),
                "risk_level": _risk_level(bmi_mean, recommended_week, reach_rate),
                "method": "nipt_bmi_grouping",
            }
        )
    return pd.DataFrame(rows).sort_values("bmi_min").reset_index(drop=True)


def _find_column(df: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in df.columns}
    for lowered_name, original in lowered.items():
        for term in terms:
            normalized = term.lower()
            if len(normalized) == 1 and lowered_name != normalized:
                continue
            if normalized in lowered_name:
                return original
    return None


def _bmi_groups(series: pd.Series) -> pd.Series:
    bmi = pd.to_numeric(series, errors="coerce")
    if bmi.nunique(dropna=True) < 4:
        bins = [-np.inf, 28.0, 32.0, np.inf]
        labels = ("normal_or_low", "overweight", "high_bmi")
        return pd.cut(bmi, bins=bins, labels=labels, include_lowest=True)
    try:
        return pd.qcut(bmi, q=min(4, bmi.nunique()), duplicates="drop")
    except ValueError:
        bins = [-np.inf, 28.0, 32.0, np.inf]
        labels = ("normal_or_low", "overweight", "high_bmi")
        return pd.cut(bmi, bins=bins, labels=labels, include_lowest=True)


def _recommended_week(part: pd.DataFrame, bmi_mean: float) -> tuple[float, float]:
    baseline = _bmi_baseline_week(bmi_mean)
    if "week" not in part or "y_concentration" not in part:
        return baseline, np.nan

    observed = part.dropna(subset=["week", "y_concentration"])
    if observed.empty:
        return baseline, np.nan
    observed = observed.sort_values("week")
    reached = observed[observed["y_concentration"] >= 0.04]
    reach_rate = float(len(reached) / max(len(observed), 1))
    if reached.empty:
        return max(baseline, float(np.ceil(observed["week"].median()))), reach_rate
    return float(np.ceil(reached["week"].quantile(0.25))), reach_rate


def _bmi_baseline_week(bmi_mean: float) -> float:
    if bmi_mean < 28.0:
        return 12.0
    if bmi_mean < 32.0:
        return 13.0
    return 14.0


def _abnormal_rate(part: pd.DataFrame) -> float:
    if "abnormal" not in part:
        return np.nan
    values = pd.to_numeric(part["abnormal"], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((values > 0).mean())


def _risk_level(bmi_mean: float, recommended_week: float, reach_rate: float) -> str:
    if bmi_mean >= 32.0 or recommended_week >= 14.0 or (np.isfinite(reach_rate) and reach_rate < 0.6):
        return "high"
    if bmi_mean >= 28.0 or recommended_week >= 13.0:
        return "medium"
    return "low"
