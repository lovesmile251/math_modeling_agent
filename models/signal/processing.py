from __future__ import annotations

import numpy as np
import pandas as pd


def fft_frequency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_signals(df)
    if numeric.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 4 or np.std(values) == 0:
            continue
        centered = values - values.mean()
        spectrum = np.fft.rfft(centered)
        frequencies = np.fft.rfftfreq(len(centered), d=1.0)
        amplitudes = np.abs(spectrum) / len(centered)
        order = np.argsort(-amplitudes)
        rank = 0
        for idx in order:
            if frequencies[idx] == 0:
                continue
            rank += 1
            rows.append(
                {
                    "signal": str(column),
                    "rank": rank,
                    "frequency": float(frequencies[idx]),
                    "period": float(1 / frequencies[idx]) if frequencies[idx] > 0 else float("inf"),
                    "amplitude": float(amplitudes[idx]),
                    "sample_size": int(len(values)),
                    "method": "fft",
                }
            )
            if rank >= 5:
                break
    return pd.DataFrame(rows)


def signal_denoising(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    numeric = _numeric_signals(df)
    if numeric.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    window = max(3, int(window))
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < window:
            continue
        smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
        residual = series - smoothed
        before_std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        after_std = float(residual.std(ddof=1)) if len(residual) > 1 else 0.0
        for idx, raw, smooth, noise in zip(series.index, series, smoothed, residual):
            rows.append(
                {
                    "row_index": idx,
                    "signal": str(column),
                    "raw_value": float(raw),
                    "denoised_value": float(smooth),
                    "estimated_noise": float(noise),
                    "window": window,
                    "raw_std": before_std,
                    "residual_std": after_std,
                    "method": "centered_moving_average_filter",
                }
            )
    return pd.DataFrame(rows)


def energy_detection(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_signals(df)
    if numeric.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if len(series) < 3:
            continue
        values = series.to_numpy(dtype=float)
        energy = values**2
        threshold = float(energy.mean() + 2 * energy.std(ddof=0))
        for idx, raw, item_energy in zip(series.index, values, energy):
            rows.append(
                {
                    "row_index": idx,
                    "signal": str(column),
                    "value": float(raw),
                    "energy": float(item_energy),
                    "threshold": threshold,
                    "detected": int(item_energy >= threshold),
                    "total_energy": float(energy.sum()),
                    "mean_energy": float(energy.mean()),
                    "method": "energy_threshold_detection",
                }
            )
    return pd.DataFrame(rows)


def _numeric_signals(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] == 0:
        return pd.DataFrame()
    candidates = []
    for column in numeric.columns:
        name = str(column).lower()
        if any(token in name for token in ("id", "index", "row")):
            continue
        if pd.to_numeric(numeric[column], errors="coerce").dropna().std() > 0:
            candidates.append(column)
    return numeric[candidates] if candidates else pd.DataFrame()
