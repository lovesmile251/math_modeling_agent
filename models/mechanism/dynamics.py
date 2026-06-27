from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic_growth_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 4 or numeric.shape[1] < 1:
        return pd.DataFrame()

    time_column = _find_column(numeric, ("time", "date", "year", "month", "period", "t", "时间", "日期", "年份", "月份", "期"))
    target_column = _find_column(
        numeric,
        ("population", "pop", "size", "quantity", "count", "demand", "y", "数量", "人口", "规模", "总量", "需求"),
        exclude={time_column} if time_column else set(),
    )
    if target_column is None:
        target_column = _first_numeric_column(numeric, exclude={time_column} if time_column else set())
    if target_column is None:
        return pd.DataFrame()

    data = _xy_data(numeric, time_column, target_column)
    if len(data) < 4 or (data["y"] <= 0).any() or np.allclose(data["y"], data["y"].iloc[0]):
        return pd.DataFrame()

    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    x0 = x.min()
    x = x - x0
    y_max = float(np.max(y))
    if y_max <= 0:
        return pd.DataFrame()

    best: tuple[float, float, float, np.ndarray, float] | None = None
    for multiplier in np.linspace(1.05, 4.0, 60):
        carrying_capacity = y_max * float(multiplier)
        transformed = carrying_capacity / y - 1.0
        if np.any(transformed <= 0):
            continue
        logit = np.log(transformed)
        try:
            slope, intercept = np.polyfit(x, logit, 1)
        except (ValueError, np.linalg.LinAlgError):
            continue
        growth_rate = -float(slope)
        initial_ratio = float(math.exp(intercept))
        fitted = carrying_capacity / (1.0 + initial_ratio * np.exp(-growth_rate * x))
        sse = float(np.sum((y - fitted) ** 2))
        if best is None or sse < best[-1]:
            best = (carrying_capacity, growth_rate, initial_ratio, fitted, sse)
    if best is None:
        return pd.DataFrame()

    carrying_capacity, growth_rate, initial_ratio, fitted, _ = best
    rmse = _rmse(y, fitted)
    return pd.DataFrame(
        [
            {
                "method": "logistic_growth_model",
                "target": str(target_column),
                "time_column": str(time_column) if time_column else "row_index",
                "carrying_capacity": float(carrying_capacity),
                "growth_rate": float(growth_rate),
                "initial_ratio": float(initial_ratio),
                "initial_value": float(y[0]),
                "rmse": rmse,
                "r_squared": _r_squared(y, fitted),
                "sample_size": int(len(y)),
            }
        ]
    )


def sir_epidemic_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 3:
        return pd.DataFrame()

    susceptible_column = _find_column(numeric, ("susceptible", "sus", "s", "易感", "易感者"))
    infected_column = _find_column(
        numeric,
        ("infected", "infectious", "active", "case", "cases", "i", "感染", "感染者", "病例", "患病"),
        exclude={susceptible_column},
    )
    recovered_column = _find_column(
        numeric,
        ("recovered", "removed", "recovery", "r", "康复", "移除", "治愈"),
        exclude={susceptible_column, infected_column},
    )
    if susceptible_column is None or infected_column is None or recovered_column is None:
        return pd.DataFrame()

    data = numeric[[susceptible_column, infected_column, recovered_column]].dropna()
    if len(data) < 3:
        return pd.DataFrame()
    s = data[susceptible_column].to_numpy(dtype=float)
    i = data[infected_column].to_numpy(dtype=float)
    r = data[recovered_column].to_numpy(dtype=float)
    if np.any(s < 0) or np.any(i < 0) or np.any(r < 0) or np.any(i[:-1] <= 0):
        return pd.DataFrame()
    population = s + i + r
    n = float(np.nanmedian(population))
    if not np.isfinite(n) or n <= 0:
        return pd.DataFrame()

    ds = np.diff(s)
    dr = np.diff(r)
    infection_term = s[:-1] * i[:-1] / n
    valid_beta = infection_term > 0
    valid_gamma = i[:-1] > 0
    if not valid_beta.any() or not valid_gamma.any():
        return pd.DataFrame()

    beta = float(np.mean((-ds[valid_beta]) / infection_term[valid_beta]))
    gamma = float(np.mean(dr[valid_gamma] / i[:-1][valid_gamma]))
    if not np.isfinite(beta) or not np.isfinite(gamma):
        return pd.DataFrame()
    predicted_ds = -beta * infection_term
    predicted_dr = gamma * i[:-1]
    predicted_di = -predicted_ds - predicted_dr
    observed_di = np.diff(i)

    return pd.DataFrame(
        [
            {
                "method": "sir_epidemic_model",
                "susceptible": str(susceptible_column),
                "infected": str(infected_column),
                "recovered": str(recovered_column),
                "beta": beta,
                "gamma": gamma,
                "basic_reproduction_number": float(beta / gamma) if gamma != 0 else np.nan,
                "population": n,
                "rmse_infected_delta": _rmse(observed_di, predicted_di),
                "rmse_susceptible_delta": _rmse(ds, predicted_ds),
                "rmse_recovered_delta": _rmse(dr, predicted_dr),
                "sample_size": int(len(data)),
            }
        ]
    )


def lotka_volterra_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 4:
        return pd.DataFrame()

    prey_column = _find_column(numeric, ("prey", "resource", "x", "rabbit", "host", "猎物", "资源", "被捕食", "兔"))
    predator_column = _find_column(numeric, ("predator", "consumer", "y", "fox", "捕食", "捕食者", "天敌", "狐"), exclude={prey_column} if prey_column else set())
    if prey_column is None or predator_column is None:
        columns = list(numeric.columns)
        if len(columns) < 2:
            return pd.DataFrame()
        prey_column = str(columns[0])
        predator_column = str(columns[1])

    data = numeric[[prey_column, predator_column]].dropna()
    if len(data) < 4:
        return pd.DataFrame()
    prey = data[prey_column].to_numpy(dtype=float)
    predator = data[predator_column].to_numpy(dtype=float)
    if np.any(prey <= 0) or np.any(predator <= 0):
        return pd.DataFrame()

    dx = np.diff(prey) / prey[:-1]
    dy = np.diff(predator) / predator[:-1]
    design_prey = np.column_stack([np.ones(len(dx)), -predator[:-1]])
    design_predator = np.column_stack([prey[:-1], -np.ones(len(dy))])
    try:
        alpha, beta = np.linalg.lstsq(design_prey, dx, rcond=None)[0]
        delta, gamma = np.linalg.lstsq(design_predator, dy, rcond=None)[0]
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    fitted_dx = design_prey @ np.array([alpha, beta])
    fitted_dy = design_predator @ np.array([delta, gamma])
    return pd.DataFrame(
        [
            {
                "method": "lotka_volterra_model",
                "prey": str(prey_column),
                "predator": str(predator_column),
                "prey_growth_alpha": float(alpha),
                "predation_beta": float(beta),
                "predator_growth_delta": float(delta),
                "predator_mortality_gamma": float(gamma),
                "rmse_prey_relative_delta": _rmse(dx, fitted_dx),
                "rmse_predator_relative_delta": _rmse(dy, fitted_dy),
                "sample_size": int(len(data)),
            }
        ]
    )


def solow_growth_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 4:
        return pd.DataFrame()

    output_column = _find_column(numeric, ("output", "gdp", "income", "production", "y", "产出", "收入", "生产", "国内生产总值"))
    capital_column = _find_column(numeric, ("capital", "asset", "investment_stock", "k", "资本", "资产", "存量"), exclude={output_column} if output_column else set())
    labor_column = _find_column(numeric, ("labor", "labour", "worker", "employment", "population", "l", "劳动", "劳动力", "就业", "人口"), exclude={output_column, capital_column})
    investment_column = _find_column(numeric, ("investment", "saving", "savings", "invest", "i", "投资", "储蓄"), exclude={output_column, capital_column, labor_column})
    if output_column is None or capital_column is None:
        return pd.DataFrame()

    columns = [column for column in (output_column, capital_column, labor_column, investment_column) if column is not None]
    data = numeric[columns].dropna()
    if len(data) < 4:
        return pd.DataFrame()
    output = data[output_column].to_numpy(dtype=float)
    capital = data[capital_column].to_numpy(dtype=float)
    if np.any(output <= 0) or np.any(capital <= 0):
        return pd.DataFrame()
    if labor_column is not None and np.all(data[labor_column].to_numpy(dtype=float) > 0):
        labor = data[labor_column].to_numpy(dtype=float)
    else:
        labor = np.ones(len(data), dtype=float)

    y_per_labor = output / labor
    k_per_labor = capital / labor
    if np.any(y_per_labor <= 0) or np.any(k_per_labor <= 0) or np.allclose(k_per_labor, k_per_labor[0]):
        return pd.DataFrame()
    x = np.log(k_per_labor)
    y = np.log(y_per_labor)
    try:
        alpha, log_a = np.polyfit(x, y, 1)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame()
    technology = float(math.exp(log_a))
    fitted = technology * np.power(k_per_labor, alpha)
    savings_rate = np.nan
    if investment_column is not None:
        investment = data[investment_column].to_numpy(dtype=float)
        valid = output != 0
        if valid.any():
            savings_rate = float(np.nanmean(investment[valid] / output[valid]))

    return pd.DataFrame(
        [
            {
                "method": "solow_growth_model",
                "output": str(output_column),
                "capital": str(capital_column),
                "labor": str(labor_column) if labor_column else "unit_labor",
                "investment": str(investment_column) if investment_column else "",
                "capital_elasticity_alpha": float(alpha),
                "technology_a": technology,
                "average_savings_rate": savings_rate,
                "rmse_output_per_labor": _rmse(y_per_labor, fitted),
                "r_squared_log": _r_squared(y, np.log(fitted)),
                "sample_size": int(len(data)),
            }
        ]
    )


def heat_conduction_1d(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 6:
        return pd.DataFrame()

    position_column = _find_column(numeric, ("position", "distance", "x", "space", "位置", "距离", "空间", "坐标"))
    time_column = _find_column(numeric, ("time", "t", "second", "minute", "时间", "时刻", "秒", "分钟"), exclude={position_column} if position_column else set())
    temperature_column = _find_column(numeric, ("temperature", "temp", "heat", "theta", "温度", "热量"), exclude={position_column, time_column})
    if position_column is None or time_column is None or temperature_column is None:
        return pd.DataFrame()

    data = numeric[[position_column, time_column, temperature_column]].dropna()
    if len(data) < 6:
        return pd.DataFrame()
    pivot = data.pivot_table(index=time_column, columns=position_column, values=temperature_column, aggfunc="mean").sort_index().sort_index(axis=1)
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return pd.DataFrame()
    times = pivot.index.to_numpy(dtype=float)
    positions = pivot.columns.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    estimates: list[float] = []
    observed: list[float] = []
    predicted_base: list[float] = []
    for ti in range(pivot.shape[0] - 1):
        dt = times[ti + 1] - times[ti]
        if dt <= 0:
            continue
        for xi in range(1, pivot.shape[1] - 1):
            dx_left = positions[xi] - positions[xi - 1]
            dx_right = positions[xi + 1] - positions[xi]
            if dx_left <= 0 or dx_right <= 0:
                continue
            t_now = values[ti, xi]
            t_next = values[ti + 1, xi]
            if not np.isfinite(t_now) or not np.isfinite(t_next):
                continue
            second_derivative = 2.0 * (
                (values[ti, xi + 1] - t_now) / dx_right - (t_now - values[ti, xi - 1]) / dx_left
            ) / (dx_left + dx_right)
            time_derivative = (t_next - t_now) / dt
            if np.isfinite(second_derivative) and abs(second_derivative) > 1e-12:
                estimates.append(time_derivative / second_derivative)
                observed.append(time_derivative)
                predicted_base.append(second_derivative)
    if len(estimates) < 2:
        return pd.DataFrame()
    alpha = float(np.median(estimates))
    observed_arr = np.array(observed, dtype=float)
    predicted_arr = alpha * np.array(predicted_base, dtype=float)
    return pd.DataFrame(
        [
            {
                "method": "heat_conduction_1d",
                "position": str(position_column),
                "time_column": str(time_column),
                "temperature": str(temperature_column),
                "thermal_diffusivity_alpha": alpha,
                "rmse_temperature_time_derivative": _rmse(observed_arr, predicted_arr),
                "grid_time_points": int(pivot.shape[0]),
                "grid_position_points": int(pivot.shape[1]),
                "sample_size": int(len(estimates)),
            }
        ]
    )


def harmonic_oscillator_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 5:
        return pd.DataFrame()

    time_column = _find_column(numeric, ("time", "t", "second", "date", "时间", "时刻", "秒", "日期"))
    displacement_column = _find_column(numeric, ("displacement", "position", "amplitude", "x", "y", "位移", "位置", "振幅"), exclude={time_column} if time_column else set())
    if displacement_column is None:
        displacement_column = _first_numeric_column(numeric, exclude={time_column} if time_column else set())
    if displacement_column is None:
        return pd.DataFrame()

    data = _xy_data(numeric, time_column, displacement_column)
    if len(data) < 5:
        return pd.DataFrame()
    t = data["x"].to_numpy(dtype=float)
    x = data["y"].to_numpy(dtype=float)
    t = t - t.min()
    span = float(t.max() - t.min())
    if span <= 0 or np.allclose(x, x[0]):
        return pd.DataFrame()
    min_omega = 2.0 * math.pi / max(span * 4.0, 1e-9)
    max_omega = 2.0 * math.pi * max((len(t) - 1) / max(2.0 * span, 1e-9), min_omega)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for omega in np.linspace(min_omega, max_omega, 200):
        design = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(len(t))])
        try:
            coefficients = np.linalg.lstsq(design, x, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        fitted = design @ coefficients
        sse = float(np.sum((x - fitted) ** 2))
        if best is None or sse < best[-1]:
            best = (float(omega), coefficients, fitted, sse)
    if best is None:
        return pd.DataFrame()

    omega, coefficients, fitted, _ = best
    cos_coef, sin_coef, offset = coefficients
    amplitude = float(math.sqrt(cos_coef**2 + sin_coef**2))
    phase = float(math.atan2(-sin_coef, cos_coef))
    return pd.DataFrame(
        [
            {
                "method": "harmonic_oscillator_model",
                "target": str(displacement_column),
                "time_column": str(time_column) if time_column else "row_index",
                "amplitude": amplitude,
                "angular_frequency": omega,
                "period": float(2.0 * math.pi / omega) if omega != 0 else np.nan,
                "phase": phase,
                "offset": float(offset),
                "rmse": _rmse(x, fitted),
                "r_squared": _r_squared(x, fitted),
                "sample_size": int(len(x)),
            }
        ]
    )


def michaelis_menten_kinetics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 3:
        return pd.DataFrame()

    substrate_column = _find_column(numeric, ("substrate", "concentration", "s", "conc", "底物", "浓度"))
    velocity_column = _find_column(numeric, ("velocity", "rate", "reaction", "v", "speed", "速率", "速度", "反应"), exclude={substrate_column} if substrate_column else set())
    if substrate_column is None or velocity_column is None:
        columns = list(numeric.columns)
        if len(columns) < 2:
            return pd.DataFrame()
        substrate_column = str(columns[0])
        velocity_column = str(columns[1])

    data = numeric[[substrate_column, velocity_column]].dropna()
    if len(data) < 3:
        return pd.DataFrame()
    substrate = data[substrate_column].to_numpy(dtype=float)
    velocity = data[velocity_column].to_numpy(dtype=float)
    valid = (substrate > 0) & (velocity > 0)
    if valid.sum() < 3:
        return pd.DataFrame()
    substrate = substrate[valid]
    velocity = velocity[valid]
    reciprocal_x = 1.0 / substrate
    reciprocal_y = 1.0 / velocity
    try:
        slope, intercept = np.polyfit(reciprocal_x, reciprocal_y, 1)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame()
    if intercept <= 0:
        return pd.DataFrame()
    vmax = float(1.0 / intercept)
    km = float(slope * vmax)
    fitted = vmax * substrate / (km + substrate)
    return pd.DataFrame(
        [
            {
                "method": "michaelis_menten_kinetics",
                "substrate": str(substrate_column),
                "velocity": str(velocity_column),
                "vmax": vmax,
                "km": km,
                "rmse_velocity": _rmse(velocity, fitted),
                "r_squared": _r_squared(velocity, fitted),
                "sample_size": int(len(substrate)),
            }
        ]
    )


def bernoulli_flow_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_frame(df)
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        return pd.DataFrame()

    velocity_column = _find_column(numeric, ("velocity", "speed", "flow_velocity", "v", "流速", "速度", "流速度"))
    pressure_column = _find_column(numeric, ("pressure", "press", "p", "压强", "压力"), exclude={velocity_column} if velocity_column else set())
    if velocity_column is None or pressure_column is None:
        return pd.DataFrame()

    height_column = _find_column(numeric, ("height", "elevation", "z", "高度", "高程", "标高"), exclude={velocity_column, pressure_column})
    density_column = _find_column(numeric, ("density", "rho", "密度"), exclude={velocity_column, pressure_column, height_column})
    diameter_column = _find_column(numeric, ("diameter", "pipe_diameter", "d", "直径", "管径"), exclude={velocity_column, pressure_column, height_column, density_column})
    viscosity_column = _find_column(numeric, ("viscosity", "mu", "黏度", "粘度"), exclude={velocity_column, pressure_column, height_column, density_column, diameter_column})

    columns = [column for column in (velocity_column, pressure_column, height_column, density_column, diameter_column, viscosity_column) if column is not None]
    data = numeric[columns].dropna()
    if len(data) < 2:
        return pd.DataFrame()

    gravity = 9.81
    velocity = data[velocity_column].to_numpy(dtype=float)
    pressure = data[pressure_column].to_numpy(dtype=float)
    if np.any(velocity < 0):
        return pd.DataFrame()
    elevation = data[height_column].to_numpy(dtype=float) if height_column else np.zeros(len(data), dtype=float)
    if density_column:
        density = data[density_column].to_numpy(dtype=float)
        density = np.where(density > 0, density, 1000.0)
    else:
        density = np.full(len(data), 1000.0, dtype=float)

    velocity_head = velocity**2 / (2.0 * gravity)
    pressure_head = pressure / (density * gravity)
    elevation_head = elevation
    total_head = velocity_head + pressure_head + elevation_head
    reference_head = float(np.max(total_head))
    head_loss = reference_head - total_head

    if diameter_column and viscosity_column:
        diameter = data[diameter_column].to_numpy(dtype=float)
        viscosity = data[viscosity_column].to_numpy(dtype=float)
        reynolds = np.where(viscosity > 0, density * velocity * diameter / viscosity, np.nan)
    else:
        reynolds = np.full(len(data), np.nan, dtype=float)

    rows: list[dict[str, float | str | int]] = []
    for idx in range(len(data)):
        re_value = float(reynolds[idx]) if np.isfinite(reynolds[idx]) else np.nan
        if np.isnan(re_value):
            regime = "unknown"
        elif re_value < 2300:
            regime = "laminar"
        elif re_value <= 4000:
            regime = "transitional"
        else:
            regime = "turbulent"
        rows.append(
            {
                "row_index": int(data.index[idx]),
                "velocity": float(velocity[idx]),
                "pressure": float(pressure[idx]),
                "elevation": float(elevation_head[idx]),
                "density": float(density[idx]),
                "velocity_head": float(velocity_head[idx]),
                "pressure_head": float(pressure_head[idx]),
                "elevation_head": float(elevation_head[idx]),
                "total_head": float(total_head[idx]),
                "head_loss_vs_max": float(head_loss[idx]),
                "reynolds_number": re_value,
                "flow_regime": regime,
                "method": "bernoulli_energy_head_analysis",
            }
        )
    return pd.DataFrame(rows)


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    return numeric.copy()


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    excluded = {str(column) for column in (exclude or set()) if column is not None}
    for column in df.columns:
        if str(column) in excluded:
            continue
        name = str(column).lower()
        if any(_keyword_matches(name, keyword) for keyword in keywords):
            return str(column)
    return None


def _keyword_matches(name: str, keyword: str) -> bool:
    key = keyword.lower()
    if len(key) == 1 and key.isascii() and key.isalpha():
        return name == key
    return key in name


def _first_numeric_column(df: pd.DataFrame, exclude: set[str | None] | None = None) -> str | None:
    excluded = {str(column) for column in (exclude or set()) if column is not None}
    for column in df.columns:
        if str(column) not in excluded:
            return str(column)
    return None


def _xy_data(df: pd.DataFrame, time_column: str | None, target_column: str) -> pd.DataFrame:
    if time_column and time_column in df.columns:
        data = pd.DataFrame({"x": df[time_column], "y": df[target_column]}).dropna()
    else:
        values = df[target_column].dropna()
        data = pd.DataFrame({"x": np.arange(len(values), dtype=float), "y": values.to_numpy(dtype=float)})
    return data


def _rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    if len(observed) == 0:
        return np.nan
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def _r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total == 0:
        return 1.0
    residual = float(np.sum((observed - fitted) ** 2))
    return float(1.0 - residual / total)
