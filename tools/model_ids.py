from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from tools.model_registry import registered_model_ids


MODEL_ID_ALIASES: dict[str, str] = {
    "correlation": "correlation_analysis",
    "correlation_analysis_model": "correlation_analysis",
    "pearson": "correlation_analysis",
    "pearson_correlation": "correlation_analysis",
    "spearman": "correlation_analysis",
    "spearman_correlation": "correlation_analysis",
    "regression": "linear_regression",
    "linear_regression_analysis": "linear_regression",
    "topsis": "topsis_rank",
    "entropy": "entropy_weights",
    "entropy_weight": "entropy_weights",
    "ahp": "ahp_weights",
    "gm11": "grey_gm11",
    "gm_1_1": "grey_gm11",
    "kmeans": "kmeans_cluster",
    "dbscan": "dbscan_cluster",
    "pca_analysis": "pca",
}


@dataclass(frozen=True)
class NormalizedModelIds:
    selected: list[str]
    dropped: list[str]


def canonical_model_id(value: object) -> str | None:
    """Return the executable model_id for a raw LLM/UI value, if known."""

    if value is None:
        return None
    raw = str(value).strip().strip("`'\"")
    if not raw:
        return None

    normalized = re.sub(r"[\s\-]+", "_", raw.lower())
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    if not normalized:
        return None

    registry = registered_model_ids()
    if normalized in registry:
        return normalized
    alias = MODEL_ID_ALIASES.get(normalized)
    if alias in registry:
        return alias
    return None


def normalize_model_ids(values: Iterable[object]) -> NormalizedModelIds:
    """Normalize, de-duplicate, and validate model IDs from external sources."""

    selected: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for value in values:
        model_id = canonical_model_id(value)
        if model_id is None:
            raw = str(value).strip()
            if raw:
                dropped.append(raw)
            continue
        if model_id not in seen:
            selected.append(model_id)
            seen.add(model_id)
    return NormalizedModelIds(selected=selected, dropped=dropped)
