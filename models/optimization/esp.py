from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "Temp_C",
    "C_in_gNm3",
    "Q_Nm3h",
    "U1_kV",
    "U2_kV",
    "U3_kV",
    "U4_kV",
    "T1_s",
    "T2_s",
    "T3_s",
    "T4_s",
    "C_out_mgNm3",
    "P_total_kW",
)
CONTROL_COLUMNS = ("U1_kV", "U2_kV", "U3_kV", "U4_kV", "T1_s", "T2_s", "T3_s", "T4_s")


def is_cement_esp_schema(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in REQUIRED_COLUMNS)


def is_esp_operating_frame(df: pd.DataFrame) -> bool:
    """Backward-compatible schema checker used by generated scripts."""

    return is_cement_esp_schema(df)


def cement_esp_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Return ESP operating set-points for cement kiln dust-control scenarios.

    The historical sample normally sits near the old 50 mg/Nm3 outlet limit.
    Lower targets are therefore explicit engineering extrapolations from a
    collection-intensity proxy, not hidden interpolation claims.
    """

    if not is_cement_esp_schema(df):
        return pd.DataFrame()

    work = df[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 20:
        return pd.DataFrame()

    profile = _profile(work)
    power_coef = _fit_power_model(work)
    rows: list[dict[str, float | str | bool]] = []
    scenarios = {
        "typical": 0.50,
        "high_concentration_high_flow": 0.90,
        "low_load": 0.10,
    }
    targets = (50.0, 30.0, 10.0, 5.0)
    for scenario, quantile in scenarios.items():
        context = _context(work, quantile)
        for target in targets:
            row = _optimize_for_target(work, context, profile, power_coef, target)
            row["scenario"] = scenario
            rows.append(row)

    result = pd.DataFrame(rows)
    ordered = [
        "scenario",
        "target_C_out_mgNm3",
        "predicted_C_out_mgNm3",
        "predicted_P_total_kW",
        "energy_increase_percent",
        "constraint_satisfied",
        "extrapolation_level",
        *CONTROL_COLUMNS,
        "Temp_C",
        "C_in_gNm3",
        "Q_Nm3h",
        "method",
    ]
    return result[ordered]


def esp_operating_optimization(
    df: pd.DataFrame,
    standards: tuple[float, float] = (10.0, 5.0),
) -> pd.DataFrame:
    """Backward-compatible submission table for the A-case ESP workflow."""

    normalized = cement_esp_optimization(df)
    if normalized.empty:
        return pd.DataFrame()

    standards = tuple(float(item) for item in standards)
    selected = normalized[normalized["target_C_out_mgNm3"].astype(float).isin(standards)].copy()
    if selected.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, item in selected.iterrows():
        rows.append(
            {
                "section": "typical_condition_optimum",
                "condition": str(item["scenario"]),
                "condition_strategy": _condition_strategy(str(item["scenario"])),
                "standard_mgNm3": float(item["target_C_out_mgNm3"]),
                "baseline_standard_mgNm3": np.nan,
                "typical_C_in_gNm3": float(item["C_in_gNm3"]),
                "typical_Temp_C": float(item["Temp_C"]),
                "typical_Q_Nm3h": float(item["Q_Nm3h"]),
                **{column: float(item[column]) for column in CONTROL_COLUMNS},
                "predicted_C_out_mgNm3": float(item["predicted_C_out_mgNm3"]),
                "predicted_P_total_kW": float(item["predicted_P_total_kW"]),
                "baseline_P_total_kW": np.nan,
                "energy_increment_pct": float(item["energy_increase_percent"]),
                "feasibility": "satisfied" if bool(item["constraint_satisfied"]) else "violated",
                "emission_margin_mgNm3": float(item["target_C_out_mgNm3"]) - float(item["predicted_C_out_mgNm3"]),
                "control_index": _control_index(item),
                "priority_rule": np.nan,
                "method": str(item["method"]),
            }
        )

    rows.extend(_standard_tightening_summary(selected, standards))
    rows.extend(_differential_strategy(selected, standards[-1]))
    result = pd.DataFrame(rows)
    return _ordered_submission_result(result)


def _condition_strategy(scenario: str) -> str:
    return {
        "typical": "median operating condition",
        "high_concentration_high_flow": "high inlet concentration and high gas flow",
        "low_load": "low inlet concentration and low gas flow",
    }.get(scenario, scenario)


def _control_index(row) -> float:
    return float(sum(float(row[f"U{i}_kV"]) * float(row[f"T{i}_s"]) for i in range(1, 5)))


def _standard_tightening_summary(selected: pd.DataFrame, standards: tuple[float, ...]) -> list[dict]:
    if len(standards) < 2:
        return []
    base_standard = float(standards[0])
    strict_standard = float(standards[-1])
    base = selected[selected["target_C_out_mgNm3"].astype(float) == base_standard]
    strict = selected[selected["target_C_out_mgNm3"].astype(float) == strict_standard]
    if base.empty or strict.empty:
        return []
    base_power = float(base["predicted_P_total_kW"].mean())
    strict_power = float(strict["predicted_P_total_kW"].mean())
    return [
        {
            "section": "standard_tightening_summary",
            "condition": "all_typical_conditions",
            "condition_strategy": "average of selected condition optima",
            "standard_mgNm3": strict_standard,
            "baseline_standard_mgNm3": base_standard,
            "predicted_P_total_kW": strict_power,
            "baseline_P_total_kW": base_power,
            "energy_increment_pct": 100.0 * (strict_power - base_power) / max(base_power, 1e-9),
            "method": "cement_esp_collection_intensity_surrogate",
        }
    ]


def _differential_strategy(selected: pd.DataFrame, strict_standard: float) -> list[dict]:
    strict = selected[selected["target_C_out_mgNm3"].astype(float) == float(strict_standard)]
    rows: list[dict] = []
    for scenario, priority_rule in (
        ("high_concentration_high_flow", "voltage_first_then_rapping"),
        ("low_load", "voltage_saving_with_rapping_guardrail"),
    ):
        subset = strict[strict["scenario"] == scenario]
        if subset.empty:
            continue
        item = subset.iloc[0]
        rows.append(
            {
                "section": "differential_strategy",
                "condition": scenario,
                "condition_strategy": _condition_strategy(scenario),
                "standard_mgNm3": float(strict_standard),
                "typical_C_in_gNm3": float(item["C_in_gNm3"]),
                "typical_Temp_C": float(item["Temp_C"]),
                "typical_Q_Nm3h": float(item["Q_Nm3h"]),
                **{column: float(item[column]) for column in CONTROL_COLUMNS},
                "predicted_C_out_mgNm3": float(item["predicted_C_out_mgNm3"]),
                "predicted_P_total_kW": float(item["predicted_P_total_kW"]),
                "energy_increment_pct": float(item["energy_increase_percent"]),
                "priority_rule": priority_rule,
                "method": "two_condition_strategy_table",
            }
        )
    return rows


def _ordered_submission_result(result: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "section",
        "condition",
        "condition_strategy",
        "standard_mgNm3",
        "baseline_standard_mgNm3",
        "typical_C_in_gNm3",
        "typical_Temp_C",
        "typical_Q_Nm3h",
        *CONTROL_COLUMNS,
        "predicted_C_out_mgNm3",
        "predicted_P_total_kW",
        "baseline_P_total_kW",
        "energy_increment_pct",
        "feasibility",
        "emission_margin_mgNm3",
        "control_index",
        "priority_rule",
        "method",
    ]
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns]


def _profile(work: pd.DataFrame) -> dict:
    med = {column: float(work[column].median()) for column in REQUIRED_COLUMNS}
    hi = {column: float(work[column].quantile(0.99)) for column in CONTROL_COLUMNS}
    lo = {column: float(work[column].quantile(0.01)) for column in CONTROL_COLUMNS}
    base_intensity = _control_intensity(med)
    return {"median": med, "high": hi, "low": lo, "base_intensity": base_intensity}


def _context(work: pd.DataFrame, quantile: float) -> dict[str, float]:
    return {
        "Temp_C": float(work["Temp_C"].quantile(quantile)),
        "C_in_gNm3": float(work["C_in_gNm3"].quantile(quantile)),
        "Q_Nm3h": float(work["Q_Nm3h"].quantile(quantile)),
    }


def _fit_power_model(work: pd.DataFrame) -> np.ndarray:
    features = [
        np.ones(len(work)),
        *[work[column].to_numpy(dtype=float) for column in CONTROL_COLUMNS],
        work["Q_Nm3h"].to_numpy(dtype=float) / 100000.0,
    ]
    x = np.column_stack(features)
    y = work["P_total_kW"].to_numpy(dtype=float)
    try:
        return np.linalg.pinv(x) @ y
    except np.linalg.LinAlgError:
        return np.array([float(work["P_total_kW"].median()), *([0.0] * (len(CONTROL_COLUMNS) + 1))])


def _optimize_for_target(
    work: pd.DataFrame,
    context: dict[str, float],
    profile: dict,
    power_coef: np.ndarray,
    target: float,
) -> dict[str, float | str | bool]:
    median = profile["median"]
    high = profile["high"]
    baseline = {**median, **context}
    baseline_power = _predict_power(baseline, power_coef)
    baseline_out = float(work["C_out_mgNm3"].median())

    best: dict[str, float | str | bool] | None = None
    for alpha in np.linspace(0.0, 1.55, 312):
        candidate = {**baseline}
        for column in CONTROL_COLUMNS:
            candidate[column] = float(median[column] + alpha * (high[column] - median[column]))
        predicted_out = _predict_outlet(baseline_out, baseline, candidate, target)
        predicted_power = _predict_power(candidate, power_coef)
        violation = max(0.0, predicted_out - target)
        objective = predicted_power + violation * 10000.0
        record = {
            "target_C_out_mgNm3": target,
            "predicted_C_out_mgNm3": round(predicted_out, 6),
            "predicted_P_total_kW": round(predicted_power, 6),
            "energy_increase_percent": round((predicted_power - baseline_power) / baseline_power * 100.0, 6),
            "constraint_satisfied": bool(predicted_out <= target * 1.001),
            "extrapolation_level": _extrapolation_level(alpha, target, work),
            "method": "cement_esp_collection_intensity_surrogate",
            "Temp_C": round(float(context["Temp_C"]), 6),
            "C_in_gNm3": round(float(context["C_in_gNm3"]), 6),
            "Q_Nm3h": round(float(context["Q_Nm3h"]), 6),
            **{column: round(float(candidate[column]), 6) for column in CONTROL_COLUMNS},
            "_objective": objective,
        }
        if best is None or float(record["_objective"]) < float(best["_objective"]):
            best = record
        if predicted_out <= target * 1.001:
            break

    assert best is not None
    best.pop("_objective", None)
    return best


def _predict_outlet(baseline_out: float, baseline: dict[str, float], candidate: dict[str, float], target: float) -> float:
    base_intensity = _control_intensity(baseline)
    candidate_intensity = _control_intensity(candidate)
    relative_gain = max(0.0, candidate_intensity / max(base_intensity, 1e-9) - 1.0)
    removal_gain = 6.2 * relative_gain
    load_penalty = 0.10 * math.log(max(candidate["Q_Nm3h"], 1.0) / max(baseline["Q_Nm3h"], 1.0))
    inlet_penalty = 0.04 * math.log(max(candidate["C_in_gNm3"], 1.0) / max(baseline["C_in_gNm3"], 1.0))
    predicted = baseline_out * math.exp(-(removal_gain - load_penalty - inlet_penalty))
    return float(max(0.1, min(predicted, baseline_out if target <= baseline_out else predicted)))


def _control_intensity(row: dict[str, float]) -> float:
    q_scale = max(float(row["Q_Nm3h"]) / 100000.0, 1e-9)
    return sum(float(row[f"U{i}_kV"]) * float(row[f"T{i}_s"]) for i in range(1, 5)) / q_scale


def _predict_power(row: dict[str, float], coef: np.ndarray) -> float:
    vector = np.array([1.0, *[float(row[column]) for column in CONTROL_COLUMNS], float(row["Q_Nm3h"]) / 100000.0])
    return float(max(0.0, vector @ coef))


def _extrapolation_level(alpha: float, target: float, work: pd.DataFrame) -> str:
    historical_min = float(work["C_out_mgNm3"].min())
    if target >= historical_min * 0.95 and alpha <= 1.0:
        return "interpolation"
    if alpha <= 1.0:
        return "moderate_extrapolation"
    return "strong_extrapolation"
