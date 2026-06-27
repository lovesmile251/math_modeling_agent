from __future__ import annotations

import numpy as np
import pandas as pd


def hierarchical_cluster(df: pd.DataFrame, n_clusters: int | None = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[0] < 2 or numeric.shape[1] == 0:
        return pd.DataFrame()

    values = numeric.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    scaled = (values - values.mean(axis=0)) / std
    n_clusters = n_clusters or min(3, max(2, int(np.sqrt(len(scaled)))))
    n_clusters = min(max(1, n_clusters), len(scaled))

    clusters: list[list[int]] = [[idx] for idx in range(len(scaled))]
    while len(clusters) > n_clusters:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _average_linkage_distance(scaled, clusters[i], clusters[j])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None:
            break
        left, right = best_pair
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]

    labels = np.zeros(len(scaled), dtype=int)
    distance_to_center = np.zeros(len(scaled), dtype=float)
    cluster_sizes = np.zeros(len(scaled), dtype=int)
    for label, members in enumerate(clusters, start=1):
        center = scaled[members].mean(axis=0)
        for member in members:
            labels[member] = label
            distance_to_center[member] = float(np.linalg.norm(scaled[member] - center))
            cluster_sizes[member] = len(members)

    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "cluster": labels,
            "cluster_size": cluster_sizes,
            "distance_to_cluster_center": distance_to_center,
            "n_clusters": n_clusters,
            "method": "agglomerative_average_linkage",
        }
    )


def _average_linkage_distance(values: np.ndarray, left: list[int], right: list[int]) -> float:
    distances = np.sqrt(((values[left, None, :] - values[None, right, :]) ** 2).sum(axis=2))
    return float(distances.mean())
