from __future__ import annotations

import numpy as np
import pandas as pd


def smote_balance_summary(df: pd.DataFrame) -> pd.DataFrame:
    label_column = _find_label_column(df)
    if label_column is None:
        return pd.DataFrame()

    numeric = df.drop(columns=[label_column]).select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame()

    work = pd.concat([numeric, df[label_column]], axis=1).dropna()
    if work.shape[0] < 4:
        return pd.DataFrame()

    labels = work[label_column].astype(str)
    classes = labels.value_counts()
    if len(classes) < 2:
        return pd.DataFrame()

    feature_values = work[numeric.columns].to_numpy(dtype=float)
    max_count = int(classes.max())
    min_count = int(classes.min())
    imbalance_ratio = float(max_count / max(min_count, 1))

    rows: list[dict[str, float | str | int]] = []
    for cls, count in classes.items():
        class_mask = labels == cls
        class_values = feature_values[class_mask.to_numpy()]
        synthetic_needed = max_count - int(count)
        synthetic = _deterministic_smote_samples(class_values, synthetic_needed)
        feasible = int(len(class_values) >= 2 and synthetic_needed > 0)
        for feature_idx, feature in enumerate(numeric.columns):
            synthetic_mean = float(synthetic[:, feature_idx].mean()) if len(synthetic) else float(class_values[:, feature_idx].mean())
            rows.append(
                {
                    "label_column": str(label_column),
                    "class": str(cls),
                    "feature": str(feature),
                    "original_count": int(count),
                    "target_count": int(max_count),
                    "synthetic_needed": int(synthetic_needed),
                    "imbalance_ratio": imbalance_ratio,
                    "smote_feasible": feasible,
                    "original_feature_mean": float(class_values[:, feature_idx].mean()),
                    "synthetic_feature_mean": synthetic_mean,
                    "method": "deterministic_smote_balance_summary",
                }
            )
    return pd.DataFrame(rows)


def _deterministic_smote_samples(values: np.ndarray, needed: int) -> np.ndarray:
    if needed <= 0 or len(values) < 2:
        return np.empty((0, values.shape[1]), dtype=float)

    samples = []
    distances = _pairwise_distances(values)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    for idx in range(needed):
        source_idx = idx % len(values)
        neighbor_idx = int(nearest[source_idx])
        fraction = ((idx % 5) + 1) / 6.0
        samples.append(values[source_idx] + fraction * (values[neighbor_idx] - values[source_idx]))
    return np.asarray(samples, dtype=float)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff**2, axis=2), 0.0))


def _find_label_column(df: pd.DataFrame) -> str | None:
    priority = ("label", "class", "target", "category", "type", "y")
    for column in df.columns:
        name = str(column).lower()
        if any(keyword == name or keyword in name for keyword in priority):
            unique_count = df[column].dropna().nunique()
            if 2 <= unique_count <= max(20, int(len(df) * 0.5)):
                return str(column)

    non_numeric = df.select_dtypes(exclude="number")
    for column in non_numeric.columns:
        unique_count = df[column].dropna().nunique()
        if 2 <= unique_count <= max(20, int(len(df) * 0.5)):
            return str(column)

    numeric = df.select_dtypes(include="number")
    for column in reversed(numeric.columns):
        unique_count = numeric[column].dropna().nunique()
        if 2 <= unique_count <= min(10, max(2, int(len(df) * 0.3))):
            return str(column)
    return None
