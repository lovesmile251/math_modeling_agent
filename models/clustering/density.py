from __future__ import annotations

import numpy as np
import pandas as pd


def dbscan_cluster(df: pd.DataFrame, eps: float | None = None, min_samples: int | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 3 or numeric.shape[1] == 0:
        return pd.DataFrame()

    values = numeric.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    scaled = (values - values.mean(axis=0)) / std
    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))

    min_samples = min_samples or max(3, min(8, int(np.sqrt(len(scaled))) + 1))
    if eps is None:
        kth = np.partition(distances, min(min_samples, len(scaled) - 1), axis=1)[:, min(min_samples, len(scaled) - 1)]
        eps = float(np.median(kth))
        if eps <= 0:
            eps = 0.5

    labels = np.full(len(scaled), fill_value=-99, dtype=int)
    cluster_id = 0
    for point in range(len(scaled)):
        if labels[point] != -99:
            continue
        neighbors = np.where(distances[point] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[point] = -1
            continue
        cluster_id += 1
        labels[point] = cluster_id
        seeds = [idx for idx in neighbors if idx != point]
        while seeds:
            current = seeds.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -99:
                continue
            labels[current] = cluster_id
            current_neighbors = np.where(distances[current] <= eps)[0].tolist()
            if len(current_neighbors) >= min_samples:
                for neighbor in current_neighbors:
                    if labels[neighbor] in {-99, -1} and neighbor not in seeds:
                        seeds.append(neighbor)

    labels[labels == -99] = -1
    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "cluster": labels,
            "is_noise": (labels == -1).astype(int),
            "eps": eps,
            "min_samples": min_samples,
            "method": "DBSCAN",
        }
    )
