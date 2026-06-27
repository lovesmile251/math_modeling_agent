from __future__ import annotations

import numpy as np
import pandas as pd


def pca_summary(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return pd.DataFrame()

    values = numeric.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    standardized = (values - values.mean(axis=0)) / std
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    eigenvalues = (singular_values**2) / max(len(values) - 1, 1)
    explained = eigenvalues / eigenvalues.sum() if eigenvalues.sum() else np.zeros_like(eigenvalues)

    component_count = min(n_components, len(explained), len(numeric.columns))
    rows: list[dict[str, float | str | int]] = []
    for component_idx in range(component_count):
        component_name = f"PC{component_idx + 1}"
        for feature, loading in zip(numeric.columns, vt[component_idx]):
            rows.append(
                {
                    "component": component_name,
                    "feature": str(feature),
                    "loading": float(loading),
                    "explained_variance_ratio": float(explained[component_idx]),
                    "cumulative_explained_variance": float(explained[: component_idx + 1].sum()),
                }
            )
    return pd.DataFrame(rows)
