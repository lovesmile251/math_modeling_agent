from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


ESP_REQUIRED_COLUMNS = {
    "Temp_C",
    "C_in_gNm3",
    "Q_Nm3h",
    "C_out_mgNm3",
    "P_total_kW",
    "U1_kV",
    "U2_kV",
    "U3_kV",
    "U4_kV",
    "T1_s",
    "T2_s",
    "T3_s",
    "T4_s",
}

U_COLS = [f"U{i}_kV" for i in range(1, 5)]
T_COLS = [f"T{i}_s" for i in range(1, 5)]
STAGE_WEIGHTS = np.array([1.15, 1.05, 0.95, 0.85], dtype=float)


def is_esp_operating_frame(df: pd.DataFrame) -> bool:
    """Return True when the table has the cement ESP control schema."""
    return ESP_REQUIRED_COLUMNS.issubset({str(column) for column in df.columns})


def esp_operating_optimization(
    df: pd.DataFrame,
    standards: tuple[float, float] = (10.0, 5.0),
) -> pd.DataFrame:
    """Optimize four-field ESP voltage/rapping settings for submission tables.

    The model combines a calibrated Deutsch-Anderson style collection response
    with a data-fitted power surrogate. It intentionally enumerates finite
    settings from observed operating quantiles and extrema so the output is an
    actionable operating table, not a symbolic optimization template.
    """
    if not is_esp_operating_frame(df):
        return pd.DataFrame()

    work = _numeric_work_frame(df)
    if work.shape[0] < 20:
        return pd.DataFrame()

    med = work.median(numeric_only=True)
    calibrator = _calibrate_collection(work, med)
    power_model = _fit_power_model(work, med)
    candidates = _control_candidates(work, med)
    if candidates.empty:
        return pd.DataFrame()

    profiles = _typical_profiles(work)
    optimum_rows: list[dict] = []
    by_profile_standard: dict[tuple[str, float], dict] = {}
    for profile in profiles:
        for standard in standards:
            best = _best_setting(
                profile=profile,
                standard=float(standard),
                candidates=candidates,
                med=med,
                calibrator=calibrator,
                power_model=power_model,
            )
            row = {
                "section": "typical_condition_optimum",
                "condition": profile["condition"],
                "condition_strategy": profile["strategy"],
                "standard_mgNm3": float(standard),
                **_profile_values(profile),
                **_setting_values(best),
            }
            by_profile_standard[(profile["condition"], float(standard))] = row
            optimum_rows.append(row)

    _attach_standard_increments(optimum_rows, standards)
    rows: list[dict] = [*_summary_rows(optimum_rows, standards), *optimum_rows]
    rows.extend(_differential_strategy_rows(profiles, by_profile_standard, standards))
    result = pd.DataFrame(rows)
    return _ordered_result(result)


def _numeric_work_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Temp_C", "C_in_gNm3", "Q_Nm3h", *U_COLS, *T_COLS, "C_out_mgNm3", "P_total_kW"]
    work = df[columns].apply(pd.to_numeric, errors="coerce")
    work = work.dropna(subset=["Temp_C", "C_in_gNm3", "Q_Nm3h", *U_COLS, *T_COLS, "P_total_kW"])
    work = work[(work["C_in_gNm3"] > 0) & (work["Q_Nm3h"] > 0) & (work[U_COLS] > 0).all(axis=1) & (work[T_COLS] > 0).all(axis=1)]
    return work.reset_index(drop=True)


def _calibrate_collection(work: pd.DataFrame, med: pd.Series) -> dict:
    valid = work.dropna(subset=["C_out_mgNm3"]).copy()
    valid = valid[valid["C_out_mgNm3"] > 0]
    if valid.empty:
        valid = work.copy()
        valid["C_out_mgNm3"] = 50.0

    voltage = _voltage_index(valid[U_COLS].to_numpy(dtype=float), med)
    rapping = _rapping_index(valid[T_COLS].to_numpy(dtype=float), med)
    severity = _severity_index(valid["Q_Nm3h"].to_numpy(dtype=float), valid["Temp_C"].to_numpy(dtype=float), med)
    inlet_mg = np.maximum(valid["C_in_gNm3"].to_numpy(dtype=float) * 1000.0, 1.0)
    outlet = np.clip(valid["C_out_mgNm3"].to_numpy(dtype=float), 0.1, None)
    observed_log_removal = np.log(inlet_mg / outlet)
    denominator = np.maximum(voltage * rapping, 1e-6)
    lambda_values = observed_log_removal * severity / denominator
    collection_lambda = float(np.nanmedian(lambda_values[np.isfinite(lambda_values)]))
    if not np.isfinite(collection_lambda) or collection_lambda <= 0:
        collection_lambda = 6.5

    return {
        "lambda": collection_lambda,
        "median_log_removal": float(np.nanmedian(observed_log_removal)),
    }


def _fit_power_model(work: pd.DataFrame, med: pd.Series) -> dict:
    voltage = _voltage_index(work[U_COLS].to_numpy(dtype=float), med)
    rapping = _rapping_index(work[T_COLS].to_numpy(dtype=float), med)
    q_norm = (work["Q_Nm3h"].to_numpy(dtype=float) - float(med["Q_Nm3h"])) / max(float(med["Q_Nm3h"]), 1.0)
    temp_norm = (work["Temp_C"].to_numpy(dtype=float) - float(med["Temp_C"])) / max(float(med["Temp_C"]), 1.0)
    x = np.column_stack(
        [
            np.ones(len(work)),
            voltage - float(np.nanmean(voltage)),
            rapping - float(np.nanmean(rapping)),
            q_norm,
            temp_norm,
        ]
    )
    y = work["P_total_kW"].to_numpy(dtype=float)
    ridge = np.diag([0.0, 1e-4, 1e-4, 1e-4, 1e-4])
    try:
        coef = np.linalg.pinv(x.T @ x + ridge) @ x.T @ y
    except np.linalg.LinAlgError:
        coef = np.array([float(np.nanmedian(y)), 0.0, 0.0, 0.0, 0.0], dtype=float)

    p10, p90 = np.nanquantile(y, [0.10, 0.90])
    spread = max(float(p90 - p10), 1.0)
    coef = np.asarray(coef, dtype=float)
    coef[1] = max(float(coef[1]), spread * 1.2)
    coef[2] = max(float(coef[2]), spread * 0.6)
    return {
        "coef": coef,
        "voltage_center": float(np.nanmean(voltage)),
        "rapping_center": float(np.nanmean(rapping)),
        "min_power": float(np.nanmin(y)),
        "max_power": float(np.nanmax(y)),
    }


def _control_candidates(work: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    u_levels = [_levels(work[column], (0.25, 0.50, 0.75, 0.90, 1.00), decimals=1) for column in U_COLS]
    t_levels = [_levels(work[column], (0.00, 0.10, 0.25, 0.50), decimals=0) for column in T_COLS]
    if any(len(levels) == 0 for levels in [*u_levels, *t_levels]):
        return pd.DataFrame()

    u_grid = np.array(list(product(*u_levels)), dtype=float)
    t_grid = np.array(list(product(*t_levels)), dtype=float)
    u_values = np.repeat(u_grid, len(t_grid), axis=0)
    t_values = np.tile(t_grid, (len(u_grid), 1))
    candidates = pd.DataFrame(
        np.column_stack([u_values, t_values]),
        columns=[*U_COLS, *T_COLS],
    )
    candidates["voltage_index"] = _voltage_index(u_values, med)
    candidates["rapping_index"] = _rapping_index(t_values, med)
    return candidates.drop_duplicates().reset_index(drop=True)


def _levels(series: pd.Series, quantiles: tuple[float, ...], decimals: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.array([], dtype=float)
    raw = [float(values.quantile(q)) for q in quantiles]
    raw.extend([float(values.min()), float(values.max())])
    rounded = np.array([round(value, decimals) for value in raw], dtype=float)
    rounded = rounded[np.isfinite(rounded) & (rounded > 0)]
    return np.unique(rounded)


def _typical_profiles(work: pd.DataFrame) -> list[dict]:
    return [
        {
            "condition": "low_concentration",
            "strategy": "low dust load: prefer voltage rollback, keep rapping near median",
            "C_in_gNm3": float(work["C_in_gNm3"].quantile(0.25)),
            "Temp_C": float(work["Temp_C"].quantile(0.50)),
            "Q_Nm3h": float(work["Q_Nm3h"].quantile(0.50)),
        },
        {
            "condition": "typical_median",
            "strategy": "median load: balance front-field voltage and regular rapping",
            "C_in_gNm3": float(work["C_in_gNm3"].quantile(0.50)),
            "Temp_C": float(work["Temp_C"].quantile(0.50)),
            "Q_Nm3h": float(work["Q_Nm3h"].quantile(0.50)),
        },
        {
            "condition": "high_concentration_high_flow",
            "strategy": "high dust load: raise voltage first, shorten rapping cycle second",
            "C_in_gNm3": float(work["C_in_gNm3"].quantile(0.75)),
            "Temp_C": float(work["Temp_C"].quantile(0.75)),
            "Q_Nm3h": float(work["Q_Nm3h"].quantile(0.75)),
        },
    ]


def _best_setting(
    *,
    profile: dict,
    standard: float,
    candidates: pd.DataFrame,
    med: pd.Series,
    calibrator: dict,
    power_model: dict,
) -> dict:
    predicted_outlet = _predict_outlet(
        c_in=float(profile["C_in_gNm3"]),
        q=float(profile["Q_Nm3h"]),
        temp=float(profile["Temp_C"]),
        voltage_index=candidates["voltage_index"].to_numpy(dtype=float),
        rapping_index=candidates["rapping_index"].to_numpy(dtype=float),
        med=med,
        calibrator=calibrator,
    )
    predicted_power = _predict_power(
        q=float(profile["Q_Nm3h"]),
        temp=float(profile["Temp_C"]),
        voltage_index=candidates["voltage_index"].to_numpy(dtype=float),
        rapping_index=candidates["rapping_index"].to_numpy(dtype=float),
        med=med,
        power_model=power_model,
    )
    feasible = predicted_outlet <= standard
    if feasible.any():
        feasible_indexes = np.where(feasible)[0]
        best_idx = int(feasible_indexes[np.argmin(predicted_power[feasible_indexes])])
        status = "feasible"
    else:
        violation = predicted_outlet / max(standard, 1e-9)
        normalized_power = predicted_power / max(float(np.nanmedian(predicted_power)), 1.0)
        best_idx = int(np.argmin(violation + 0.05 * normalized_power))
        status = "least_violation"

    row = candidates.iloc[best_idx].to_dict()
    row.update(
        {
            "predicted_C_out_mgNm3": float(predicted_outlet[best_idx]),
            "predicted_P_total_kW": float(predicted_power[best_idx]),
            "feasibility": status,
            "emission_margin_mgNm3": float(standard - predicted_outlet[best_idx]),
            "control_index": float(candidates.iloc[best_idx]["voltage_index"] * candidates.iloc[best_idx]["rapping_index"]),
            "method": "calibrated_esp_grid_search",
        }
    )
    return row


def _predict_outlet(
    *,
    c_in: float,
    q: float,
    temp: float,
    voltage_index: np.ndarray,
    rapping_index: np.ndarray,
    med: pd.Series,
    calibrator: dict,
) -> np.ndarray:
    severity = _severity_index(np.asarray([q], dtype=float), np.asarray([temp], dtype=float), med)[0]
    exponent = -float(calibrator["lambda"]) * voltage_index * rapping_index / max(severity, 1e-6)
    return c_in * 1000.0 * np.exp(exponent)


def _predict_power(
    *,
    q: float,
    temp: float,
    voltage_index: np.ndarray,
    rapping_index: np.ndarray,
    med: pd.Series,
    power_model: dict,
) -> np.ndarray:
    coef = np.asarray(power_model["coef"], dtype=float)
    q_norm = (q - float(med["Q_Nm3h"])) / max(float(med["Q_Nm3h"]), 1.0)
    temp_norm = (temp - float(med["Temp_C"])) / max(float(med["Temp_C"]), 1.0)
    power = (
        coef[0]
        + coef[1] * (voltage_index - float(power_model["voltage_center"]))
        + coef[2] * (rapping_index - float(power_model["rapping_center"]))
        + coef[3] * q_norm
        + coef[4] * temp_norm
    )
    lower = float(power_model["min_power"]) * 0.85
    upper = float(power_model["max_power"]) * 1.25
    return np.clip(power, lower, upper)


def _voltage_index(values: np.ndarray, med: pd.Series) -> np.ndarray:
    refs = np.maximum(med[U_COLS].to_numpy(dtype=float), 1.0)
    return ((values / refs) ** 2 * STAGE_WEIGHTS).sum(axis=1) / STAGE_WEIGHTS.sum()


def _rapping_index(values: np.ndarray, med: pd.Series) -> np.ndarray:
    refs = np.maximum(med[T_COLS].to_numpy(dtype=float), 1.0)
    return (np.sqrt(refs / np.maximum(values, 1.0)) * STAGE_WEIGHTS).sum(axis=1) / STAGE_WEIGHTS.sum()


def _severity_index(q: np.ndarray, temp: np.ndarray, med: pd.Series) -> np.ndarray:
    q_ref = max(float(med["Q_Nm3h"]), 1.0)
    temp_ref = max(float(med["Temp_C"]), 1.0)
    return (q / q_ref) ** 0.35 * (temp / temp_ref) ** 0.15


def _profile_values(profile: dict) -> dict:
    return {
        "typical_C_in_gNm3": float(profile["C_in_gNm3"]),
        "typical_Temp_C": float(profile["Temp_C"]),
        "typical_Q_Nm3h": float(profile["Q_Nm3h"]),
    }


def _setting_values(setting: dict) -> dict:
    fields = [*U_COLS, *T_COLS]
    return {field: float(setting[field]) for field in fields} | {
        "predicted_C_out_mgNm3": float(setting["predicted_C_out_mgNm3"]),
        "predicted_P_total_kW": float(setting["predicted_P_total_kW"]),
        "energy_increment_pct": np.nan,
        "feasibility": str(setting["feasibility"]),
        "emission_margin_mgNm3": float(setting["emission_margin_mgNm3"]),
        "control_index": float(setting["control_index"]),
        "method": str(setting["method"]),
    }


def _attach_standard_increments(rows: list[dict], standards: tuple[float, float]) -> None:
    if len(standards) < 2:
        return
    base_standard, strict_standard = float(standards[0]), float(standards[1])
    by_condition = {(row["condition"], float(row["standard_mgNm3"])): row for row in rows}
    for row in rows:
        if float(row["standard_mgNm3"]) != strict_standard:
            continue
        base = by_condition.get((row["condition"], base_standard))
        if not base:
            continue
        base_power = float(base["predicted_P_total_kW"])
        row["energy_increment_pct"] = 100.0 * (float(row["predicted_P_total_kW"]) - base_power) / max(base_power, 1e-9)


def _summary_rows(rows: list[dict], standards: tuple[float, float]) -> list[dict]:
    if len(standards) < 2:
        return []
    base_standard, strict_standard = float(standards[0]), float(standards[1])
    base = [row for row in rows if float(row["standard_mgNm3"]) == base_standard]
    strict = [row for row in rows if float(row["standard_mgNm3"]) == strict_standard]
    if not base or not strict:
        return []
    base_power = float(np.mean([row["predicted_P_total_kW"] for row in base]))
    strict_power = float(np.mean([row["predicted_P_total_kW"] for row in strict]))
    increment = 100.0 * (strict_power - base_power) / max(base_power, 1e-9)
    return [
        {
            "section": "standard_tightening_summary",
            "condition": "all_typical_conditions",
            "condition_strategy": "average of typical condition optima",
            "standard_mgNm3": strict_standard,
            "predicted_P_total_kW": strict_power,
            "baseline_standard_mgNm3": base_standard,
            "baseline_P_total_kW": base_power,
            "energy_increment_pct": increment,
            "method": "calibrated_esp_grid_search",
        }
    ]


def _differential_strategy_rows(
    profiles: list[dict],
    by_profile_standard: dict[tuple[str, float], dict],
    standards: tuple[float, float],
) -> list[dict]:
    strict_standard = float(standards[-1])
    selected = [profiles[0], profiles[-1]]
    rows: list[dict] = []
    for profile in selected:
        optimum = by_profile_standard.get((profile["condition"], strict_standard))
        if not optimum:
            continue
        is_high = profile["condition"].startswith("high")
        rows.append(
            {
                "section": "differential_strategy",
                "condition": profile["condition"],
                "condition_strategy": (
                    "raise U1-U2/U3-U4 before shortening T when inlet concentration and flow are high"
                    if is_high
                    else "roll back voltage first and keep T near median when inlet concentration is low"
                ),
                "standard_mgNm3": strict_standard,
                **_profile_values(profile),
                **{field: optimum.get(field, np.nan) for field in [*U_COLS, *T_COLS]},
                "predicted_C_out_mgNm3": optimum.get("predicted_C_out_mgNm3", np.nan),
                "predicted_P_total_kW": optimum.get("predicted_P_total_kW", np.nan),
                "energy_increment_pct": optimum.get("energy_increment_pct", np.nan),
                "priority_rule": "voltage_first_then_rapping" if is_high else "voltage_saving_with_rapping_guardrail",
                "method": "two_condition_strategy_table",
            }
        )
    return rows


def _ordered_result(result: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "section",
        "condition",
        "condition_strategy",
        "standard_mgNm3",
        "baseline_standard_mgNm3",
        "typical_C_in_gNm3",
        "typical_Temp_C",
        "typical_Q_Nm3h",
        *U_COLS,
        *T_COLS,
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
