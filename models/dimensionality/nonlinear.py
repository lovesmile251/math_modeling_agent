from __future__ import annotations

import numpy as np
import pandas as pd


def nonlinear_embedding(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return pd.DataFrame()

    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    values = numeric.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=0)
    usable = std > 0
    if usable.sum() < 2:
        return pd.DataFrame()
    values = values[:, usable]
    std = std[usable]
    standardized = (values - values.mean(axis=0)) / std

    if len(standardized) <= 300:
        embedding = _classical_mds(standardized)
        method = "classical_mds_distance_embedding"
    else:
        embedding = _deterministic_random_projection(standardized)
        method = "deterministic_random_projection_embedding"
    if embedding is None:
        return pd.DataFrame()

    stress = _distance_stress(standardized, embedding)
    return pd.DataFrame(
        {
            "row_index": numeric.index,
            "embedding_1": embedding[:, 0],
            "embedding_2": embedding[:, 1],
            "distance_stress": stress,
            "sample_size": int(len(embedding)),
            "feature_count": int(usable.sum()),
            "method": method,
        }
    )


def _classical_mds(values: np.ndarray) -> np.ndarray | None:
    diff = values[:, None, :] - values[None, :, :]
    distances_squared = np.sum(diff**2, axis=2)
    n = distances_squared.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ distances_squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-12
    if positive.sum() == 0:
        return None
    component_count = min(2, int(positive.sum()))
    coords = eigenvectors[:, :component_count] * np.sqrt(np.maximum(eigenvalues[:component_count], 0.0))
    if component_count == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords.astype(float)


def _deterministic_random_projection(values: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20240620)
    projection = rng.normal(size=(values.shape[1], 2))
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-12)
    embedding = values @ projection
    embedding -= embedding.mean(axis=0)
    return embedding.astype(float)


def _distance_stress(original: np.ndarray, embedding: np.ndarray) -> float:
    original_dist = _pairwise_distances(original)
    embedded_dist = _pairwise_distances(embedding)
    denom = float(np.sum(original_dist**2))
    if denom <= 0:
        return 0.0
    scale = float((original_dist * embedded_dist).sum() / max((embedded_dist**2).sum(), 1e-12))
    residual = original_dist - embedded_dist * scale
    return float(np.sqrt(np.sum(residual**2) / denom))


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff**2, axis=2), 0.0))
