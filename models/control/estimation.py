from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kalman_state_estimation(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] < 1:
        return pd.DataFrame()

    measurement_col = _find_column(numeric, ("measurement", "observed", "sensor", "value", "position", "state"))
    if measurement_col is None:
        measurement_col = str(numeric.columns[0])
    control_col = _find_column(numeric, ("control", "input", "u", "action"), exclude={measurement_col})

    measurements = pd.to_numeric(numeric[measurement_col], errors="coerce").dropna()
    if len(measurements) < 3:
        return pd.DataFrame()
    controls = (
        pd.to_numeric(numeric.loc[measurements.index, control_col], errors="coerce").fillna(0.0)
        if control_col
        else pd.Series(0.0, index=measurements.index)
    )
    diffs = measurements.diff().dropna()
    process_var = float(max(diffs.var(ddof=1) * 0.05, 1e-6)) if len(diffs) > 1 else 1e-4
    measurement_var = float(max(measurements.var(ddof=1) * 0.1, 1e-6))

    estimate = float(measurements.iloc[0])
    covariance = measurement_var
    rows = []
    for step, (idx, measurement) in enumerate(measurements.items()):
        control = float(controls.loc[idx]) if idx in controls.index else 0.0
        prior = estimate + control
        prior_covariance = covariance + process_var
        gain = prior_covariance / (prior_covariance + measurement_var)
        residual = float(measurement) - prior
        estimate = prior + gain * residual
        covariance = (1.0 - gain) * prior_covariance
        rows.append(
            {
                "step": step,
                "row_index": idx,
                "measurement": float(measurement),
                "control_input": control,
                "prior_estimate": prior,
                "state_estimate": float(estimate),
                "covariance": float(covariance),
                "kalman_gain": float(gain),
                "residual": float(residual),
                "measurement_column": str(measurement_col),
                "method": "scalar_kalman_filter",
            }
        )
    return pd.DataFrame(rows)


def optimal_control_dp(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        return pd.DataFrame()

    state_col = _find_column(numeric, ("state", "position", "level", "value", "x"))
    target_col = _find_column(numeric, ("target", "setpoint", "goal", "reference"), exclude={state_col})
    if state_col is None:
        state_col = str(numeric.columns[0])

    states = pd.to_numeric(numeric[state_col], errors="coerce").dropna()
    if len(states) < 2:
        return pd.DataFrame()
    if target_col is not None:
        target_series = pd.to_numeric(numeric.loc[states.index, target_col], errors="coerce").fillna(states.median())
    else:
        target_series = pd.Series(float(states.median()), index=states.index)

    horizon = min(len(states), 60)
    initial_state = float(states.iloc[0])
    target_values = target_series.iloc[:horizon].to_numpy(dtype=float)
    spread = float(max(states.max() - states.min(), np.std(states.to_numpy(dtype=float)), 1.0))
    action_step = spread / 8.0
    actions = np.array([-2, -1, 0, 1, 2], dtype=float) * action_step
    grid_min = min(float(states.min()), float(target_series.min()), initial_state) - spread
    grid_max = max(float(states.max()), float(target_series.max()), initial_state) + spread
    grid = np.linspace(grid_min, grid_max, 121)
    control_penalty = 0.05

    value_next = (grid - target_values[-1]) ** 2
    policy: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for t in range(horizon - 1, -1, -1):
        target = target_values[t]
        best_values = np.full(len(grid), np.inf)
        best_actions = np.zeros(len(grid), dtype=float)
        for action in actions:
            next_state = np.clip(grid + action, grid_min, grid_max)
            future = np.interp(next_state, grid, value_next)
            cost = (grid - target) ** 2 + control_penalty * action**2 + future
            improved = cost < best_values
            best_values[improved] = cost[improved]
            best_actions[improved] = action
        policy.insert(0, best_actions)
        values.insert(0, best_values)
        value_next = best_values

    rows = []
    current_state = initial_state
    for t in range(horizon):
        action = float(np.interp(current_state, grid, policy[t]))
        target = float(target_values[t])
        next_state = current_state + action
        stage_cost = (current_state - target) ** 2 + control_penalty * action**2
        rows.append(
            {
                "step": t,
                "state": float(current_state),
                "target": target,
                "control_action": action,
                "next_state": float(next_state),
                "stage_cost": float(stage_cost),
                "value_to_go": float(np.interp(current_state, grid, values[t])),
                "state_column": str(state_col),
                "method": "finite_horizon_dynamic_programming",
            }
        )
        current_state = next_state
    return pd.DataFrame(rows)


def robust_control_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        return pd.DataFrame()

    state_col = _find_column(numeric, ("state", "output", "position", "level", "value", "y"))
    target_col = _find_column(numeric, ("target", "setpoint", "goal", "reference"), exclude={state_col})
    control_col = _find_column(numeric, ("control", "input", "u", "action"), exclude={state_col, target_col})
    disturbance_col = _find_column(numeric, ("disturbance", "noise", "shock", "uncertainty"), exclude={state_col, target_col, control_col})
    if state_col is None:
        state_col = str(numeric.columns[0])

    state = pd.to_numeric(numeric[state_col], errors="coerce")
    target = pd.to_numeric(numeric[target_col], errors="coerce") if target_col else pd.Series(float(state.median()), index=numeric.index)
    aligned = pd.DataFrame({"state": state, "target": target}).dropna()
    if len(aligned) < 2:
        return pd.DataFrame()

    error = aligned["state"] - aligned["target"]
    control = pd.to_numeric(numeric.loc[aligned.index, control_col], errors="coerce").fillna(0.0) if control_col else pd.Series(0.0, index=aligned.index)
    disturbance = (
        pd.to_numeric(numeric.loc[aligned.index, disturbance_col], errors="coerce").fillna(0.0)
        if disturbance_col
        else aligned["state"].diff().fillna(0.0) - control
    )

    error_std = float(error.std(ddof=1)) if len(error) > 1 else 0.0
    disturbance_std = float(disturbance.std(ddof=1)) if len(disturbance) > 1 else 0.0
    control_effort = float(np.mean(np.square(control.to_numpy(dtype=float))))
    gain_sensitivity = float(control.std(ddof=1) / error_std) if error_std > 1e-12 and len(control) > 1 else 0.0
    robustness_index = 1.0 / (1.0 + error_std + disturbance_std + math.sqrt(max(control_effort, 0.0)) * 0.1)

    return pd.DataFrame(
        [
            {
                "state_column": str(state_col),
                "target_column": str(target_col) if target_col else "",
                "control_column": str(control_col) if control_col else "",
                "disturbance_column": str(disturbance_col) if disturbance_col else "estimated_from_state_change",
                "sample_size": int(len(aligned)),
                "mean_error": float(error.mean()),
                "max_abs_error": float(error.abs().max()),
                "error_std": error_std,
                "control_effort": control_effort,
                "disturbance_std": disturbance_std,
                "gain_sensitivity": gain_sensitivity,
                "robustness_index": float(robustness_index),
                "method": "robust_control_descriptive_summary",
            }
        ]
    )


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    exclude = exclude or set()
    for column in df.columns:
        if str(column) in exclude:
            continue
        name = str(column).lower()
        if any(keyword == name or keyword in name for keyword in keywords):
            return str(column)
    return None
