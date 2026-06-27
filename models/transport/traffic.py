from __future__ import annotations

import math

import numpy as np
import pandas as pd


def traffic_flow_cellular(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        return pd.DataFrame()

    position_col = _find_column(numeric, ("position", "pos", "cell", "location", "x"))
    speed_col = _find_column(numeric, ("speed", "velocity", "v"), exclude={position_col})
    vehicle_col = _find_column(df, ("vehicle", "car", "id"))
    if position_col is None:
        position_col = str(numeric.columns[0])
    if speed_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != position_col]
        speed_col = candidates[0] if candidates else None

    work = pd.DataFrame(
        {
            "row_index": df.index,
            "vehicle": df[vehicle_col].astype(str) if vehicle_col else [f"vehicle_{idx}" for idx in df.index],
            "position": pd.to_numeric(numeric[position_col], errors="coerce"),
            "speed": pd.to_numeric(numeric[speed_col], errors="coerce") if speed_col else 0.0,
        }
    ).dropna(subset=["position", "speed"])
    if len(work) < 2:
        return pd.DataFrame()

    work = work.sort_values("position").reset_index(drop=True)
    positions = np.rint(work["position"].to_numpy(dtype=float)).astype(int)
    speeds = np.maximum(0, np.rint(work["speed"].to_numpy(dtype=float)).astype(int))
    road_length_values = _numeric_column_values(df, ("road_length", "length", "cells"))
    road_length = int(np.nanmax(road_length_values)) if road_length_values.size else int(max(positions.max() + len(positions) + 1, len(positions) * 5))
    if road_length <= len(work):
        road_length = len(work) * 5
    vmax_values = _numeric_column_values(df, ("vmax", "max_speed", "speed_limit"))
    vmax = int(max(1, np.nanmax(vmax_values))) if vmax_values.size else int(max(1, speeds.max() + 1, 5))

    gaps = []
    next_speeds = []
    next_positions = []
    for idx, position in enumerate(positions):
        leader_position = positions[(idx + 1) % len(positions)]
        raw_gap = leader_position - position - 1 if idx < len(positions) - 1 else road_length - position + positions[0] - 1
        gap = max(int(raw_gap), 0)
        speed = min(int(speeds[idx]) + 1, vmax, gap)
        if gap <= 1:
            speed = 0
        next_position = int((position + speed) % road_length)
        gaps.append(gap)
        next_speeds.append(speed)
        next_positions.append(next_position)

    density = float(len(work) / road_length)
    result = work.copy()
    result["cell_position"] = positions
    result["speed"] = speeds
    result["gap"] = gaps
    result["next_speed"] = next_speeds
    result["next_position"] = next_positions
    result["road_length"] = road_length
    result["density"] = density
    result["flow"] = density * float(np.mean(next_speeds))
    result["position_column"] = str(position_col)
    result["method"] = "deterministic_cellular_traffic_step"
    return result[
        [
            "row_index",
            "vehicle",
            "position_column",
            "cell_position",
            "speed",
            "gap",
            "next_speed",
            "next_position",
            "road_length",
            "density",
            "flow",
            "method",
        ]
    ]


def car_following_model(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        return pd.DataFrame()

    position_col = _find_column(numeric, ("position", "pos", "location", "x"))
    speed_col = _find_column(numeric, ("speed", "velocity", "v"), exclude={position_col})
    vehicle_col = _find_column(df, ("vehicle", "car", "id"))
    length_col = _find_column(numeric, ("length", "vehicle_length", "size"), exclude={position_col, speed_col})
    if position_col is None:
        position_col = str(numeric.columns[0])
    if speed_col is None:
        candidates = [str(column) for column in numeric.columns if str(column) != position_col]
        speed_col = candidates[0] if candidates else None

    work = pd.DataFrame(
        {
            "row_index": df.index,
            "vehicle": df[vehicle_col].astype(str) if vehicle_col else [f"vehicle_{idx}" for idx in df.index],
            "position": pd.to_numeric(numeric[position_col], errors="coerce"),
            "speed": pd.to_numeric(numeric[speed_col], errors="coerce") if speed_col else 0.0,
            "vehicle_length": pd.to_numeric(numeric[length_col], errors="coerce").fillna(5.0) if length_col else 5.0,
        }
    ).dropna(subset=["position", "speed"])
    if len(work) < 2:
        return pd.DataFrame()

    work = work.sort_values("position").reset_index(drop=True)
    desired_speed = _scalar_from_columns(df, ("desired_speed", "free_speed", "speed_limit", "v0"), default=max(float(work["speed"].max()) * 1.4, 1.0))
    max_acceleration = _scalar_from_columns(df, ("max_acceleration", "acceleration", "a"), default=1.2)
    comfortable_braking = _scalar_from_columns(df, ("braking", "deceleration", "b"), default=2.0)
    min_gap = _scalar_from_columns(df, ("min_gap", "jam_gap", "s0"), default=2.0)
    time_headway = _scalar_from_columns(df, ("time_headway", "headway", "t"), default=1.5)
    dt = _scalar_from_columns(df, ("dt", "time_step"), default=1.0)
    road_length = _scalar_from_columns(df, ("road_length", "length"), default=0.0)

    rows = []
    positions = work["position"].to_numpy(dtype=float)
    speeds = np.maximum(work["speed"].to_numpy(dtype=float), 0.0)
    lengths = np.maximum(work["vehicle_length"].to_numpy(dtype=float), 0.0)
    for idx, row in work.iterrows():
        if idx < len(work) - 1:
            leader_idx = idx + 1
            gap = positions[leader_idx] - positions[idx] - lengths[leader_idx]
        elif road_length > positions[-1]:
            leader_idx = 0
            gap = road_length - positions[idx] + positions[0] - lengths[leader_idx]
        else:
            leader_idx = None
            gap = float("inf")

        speed = speeds[idx]
        leader_speed = speeds[leader_idx] if leader_idx is not None else desired_speed
        delta_v = speed - leader_speed
        safe_gap = min_gap + speed * time_headway + speed * delta_v / (2.0 * math.sqrt(max(max_acceleration * comfortable_braking, 1e-9)))
        safe_gap = max(min_gap, safe_gap)
        interaction = 0.0 if not math.isfinite(gap) else (safe_gap / max(gap, 1e-6)) ** 2
        acceleration = max_acceleration * (1.0 - (speed / max(desired_speed, 1e-6)) ** 4 - interaction)
        next_speed = max(0.0, speed + acceleration * dt)
        next_position = positions[idx] + next_speed * dt
        if road_length > 0:
            next_position %= road_length

        rows.append(
            {
                "row_index": row["row_index"],
                "vehicle": str(row["vehicle"]),
                "leader": str(work.iloc[leader_idx]["vehicle"]) if leader_idx is not None else "",
                "position": float(positions[idx]),
                "speed": float(speed),
                "gap": float(gap) if math.isfinite(gap) else float("nan"),
                "relative_speed": float(delta_v),
                "acceleration": float(acceleration),
                "next_speed": float(next_speed),
                "next_position": float(next_position),
                "desired_speed": float(desired_speed),
                "time_step": float(dt),
                "method": "intelligent_driver_car_following",
            }
        )
    return pd.DataFrame(rows)


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...], exclude: set[str | None] | None = None) -> str | None:
    exclude = exclude or set()
    for column in df.columns:
        if str(column) in exclude:
            continue
        name = str(column).lower()
        if any(keyword == name or keyword in name for keyword in keywords):
            return str(column)
    return None


def _numeric_column_values(df: pd.DataFrame, keywords: tuple[str, ...]) -> np.ndarray:
    column = _find_column(df, keywords)
    if column is None:
        return np.array([])
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)


def _scalar_from_columns(df: pd.DataFrame, keywords: tuple[str, ...], default: float) -> float:
    values = _numeric_column_values(df, keywords)
    if values.size == 0:
        return float(default)
    value = float(values[0])
    return value if math.isfinite(value) and value > 0 else float(default)
