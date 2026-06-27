from __future__ import annotations

import numpy as np
import pandas as pd


def kmeans_cluster(df: pd.DataFrame, k: int | None = None, max_iter: int = 100) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 2 or numeric.shape[1] == 0:
        return pd.DataFrame()

    k = k or min(3, max(2, int(np.sqrt(len(numeric)))))
    k = min(k, len(numeric))
    values = numeric.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    scaled = (values - values.mean(axis=0)) / std

    centroids = scaled[:k].copy()
    labels = np.zeros(len(scaled), dtype=int)
    for _ in range(max_iter):
        distances = ((scaled[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centroids[cluster_id] = scaled[mask].mean(axis=0)

    distance_to_center = np.sqrt(((scaled - centroids[labels]) ** 2).sum(axis=1))
    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "cluster": labels + 1,
            "distance_to_center": distance_to_center,
            "k": k,
        }
    )
