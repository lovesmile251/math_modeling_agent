from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import get_close_matches

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
    "esp": "cement_esp_optimization",
    "cement_esp": "cement_esp_optimization",
    "electrostatic_precipitator": "cement_esp_optimization",
    "electrostatic_precipitator_optimization": "cement_esp_optimization",
    "nipt": "nipt_bmi_grouping",
    "bmi_grouping": "nipt_bmi_grouping",
    "nipt_grouping": "nipt_bmi_grouping",
    "crop_planting": "crop_planting_plan",
    "planting_plan": "crop_planting_plan",
    "farmland_plan": "crop_planting_plan",
}


@dataclass(frozen=True)
class NormalizedModelIds:
    selected: list[str]
    dropped: list[str]


@dataclass(frozen=True)
class NormalizedModelDecision:
    selected: list[str]
    primary: str
    baseline: str
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
    close = get_close_matches(normalized, registry, n=1, cutoff=0.92)
    if close:
        return close[0]
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


def normalize_model_decision(
    *,
    selected_model_ids: Iterable[object],
    primary_model_id: object = "",
    baseline_model_id: object = "",
) -> NormalizedModelDecision:
    """Normalize a full model decision and keep all IDs executable.

    This is intentionally the single reconciliation point for LLM output,
    user edits, and downstream workflow state. Unknown IDs are dropped instead
    of being allowed to crash formulation or execution.
    """

    normalized = normalize_model_ids(selected_model_ids)
    selected = list(normalized.selected)
    dropped = list(normalized.dropped)

    primary = canonical_model_id(primary_model_id)
    raw_primary = str(primary_model_id or "").strip()
    if primary is None and raw_primary:
        dropped.append(raw_primary)

    baseline = canonical_model_id(baseline_model_id)
    raw_baseline = str(baseline_model_id or "").strip()
    if baseline is None and raw_baseline:
        dropped.append(raw_baseline)

    if primary and primary not in selected:
        selected.insert(0, primary)
    if baseline and baseline not in selected:
        selected.append(baseline)

    if not primary and selected:
        primary = selected[0]
    if baseline == primary:
        baseline = ""
    if not baseline:
        baseline = next((model_id for model_id in selected if model_id != primary), "")

    return NormalizedModelDecision(
        selected=selected,
        primary=primary or "",
        baseline=baseline or "",
        dropped=list(dict.fromkeys(dropped)),
    )
