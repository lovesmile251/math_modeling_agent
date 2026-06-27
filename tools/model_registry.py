"""Model registry: maps model_id to import/call metadata for code generation.

Each entry describes how to import and call a model function so the
CodingAgent can build baseline_analysis.py dynamically instead of from
a monolithic f-string template.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from importlib import import_module

# Registry entry: (import_module, import_name, call_template, table_suffix)
# call_template uses {var} for the result variable name, e.g. "{var} = linear_trend_forecast(df, periods=3)"
ModelEntry = tuple[str, str, str, str]

# ── basic models (each has its own if/else block) ──
BASIC_MODEL_REGISTRY: dict[str, ModelEntry] = {
    # prediction
    "trend_forecast": (
        "models.prediction.trend", "linear_trend_forecast",
        "{var} = linear_trend_forecast(df, periods=3)", "trend_forecast",
    ),
    "grey_gm11": (
        "models.prediction.grey_gm", "grey_gm11_forecast",
        "{var} = grey_gm11_forecast(df, periods=3)", "grey_gm11",
    ),
    "smoothing_forecast": (
        "models.prediction.smoothing", "smoothing_forecast",
        "{var} = smoothing_forecast(df, periods=3)", "smoothing_forecast",
    ),
    # evaluation
    "entropy_weights": (
        "models.evaluation.entropy_weight", "entropy_weights",
        "{var} = entropy_weights(df)", "entropy_weights",
    ),
    "topsis_rank": (
        "models.evaluation.topsis", "topsis_rank",
        "{var} = topsis_rank(df)", "topsis_rank",
    ),
    "ahp_weights": (
        "models.evaluation.ahp", "ahp_weights",
        "{var} = ahp_weights(df)", "ahp_weights",
    ),
    "grey_relation": (
        "models.evaluation.grey_relation", "grey_relation_rank",
        "{var} = grey_relation_rank(df)", "grey_relation",
    ),
    "vikor": (
        "models.evaluation.vikor", "vikor_rank",
        "{var} = vikor_rank(df)", "vikor",
    ),
    # diagnostics
    "capacity_gap": (
        "models.diagnostics.capacity", "demand_capacity_gap",
        "{var} = demand_capacity_gap(df)", "capacity_gap",
    ),
    # optimization
    "resource_allocation": (
        "models.optimization.resource_allocation", "resource_allocation_plan",
        "{var} = resource_allocation_plan(df)", "resource_allocation",
    ),
    "knapsack_01": (
        "models.optimization.discrete", "knapsack_01_plan",
        "{var} = knapsack_01_plan(df)", "knapsack_01",
    ),
    "assignment_plan": (
        "models.optimization.discrete", "assignment_plan",
        "{var} = assignment_plan(df)", "assignment_plan",
    ),
    "bin_packing": (
        "models.optimization.discrete", "bin_packing_plan",
        "{var} = bin_packing_plan(df)", "bin_packing",
    ),
    "scheduling_plan": (
        "models.optimization.discrete", "scheduling_plan",
        "{var} = scheduling_plan(df)", "scheduling_plan",
    ),
    # statistics
    "correlation_analysis": (
        "models.statistics.correlation", "correlation_analysis",
        "{var} = correlation_analysis(df)", "correlation_analysis",
    ),
    "linear_regression": (
        "models.statistics.regression", "linear_regression_summary",
        "{var} = linear_regression_summary(df)", "linear_regression",
    ),
    "parameter_estimation": (
        "models.statistics.estimation", "parameter_estimation",
        "{var} = parameter_estimation(df)", "parameter_estimation",
    ),
    "hypothesis_tests": (
        "models.statistics.hypothesis", "hypothesis_tests",
        "{var} = hypothesis_tests(df)", "hypothesis_tests",
    ),
    "anova_analysis": (
        "models.statistics.anova", "anova_analysis",
        "{var} = anova_analysis(df)", "anova_analysis",
    ),
    # fitting
    "polynomial_fit": (
        "models.fitting.polynomial", "polynomial_fit",
        "{var} = polynomial_fit(df)", "polynomial_fit",
    ),
    # dimensionality
    "pca": (
        "models.dimensionality.pca", "pca_summary",
        "{var} = pca_summary(df)", "pca",
    ),
    "feature_selection": (
        "models.dimensionality.feature_selection", "feature_selection",
        "{var} = feature_selection(df)", "feature_selection",
    ),
    # clustering
    "kmeans_cluster": (
        "models.clustering.kmeans", "kmeans_cluster",
        "{var} = kmeans_cluster(df)", "kmeans_cluster",
    ),
    "dbscan_cluster": (
        "models.clustering.density", "dbscan_cluster",
        "{var} = dbscan_cluster(df)", "dbscan_cluster",
    ),
    "hierarchical_cluster": (
        "models.clustering.hierarchical", "hierarchical_cluster",
        "{var} = hierarchical_cluster(df)", "hierarchical_cluster",
    ),
    # classification
    "logistic_classifier": (
        "models.classification.basic", "logistic_binary_classifier",
        "{var} = logistic_binary_classifier(df)", "logistic_classifier",
    ),
    "naive_bayes_classifier": (
        "models.classification.basic", "naive_bayes_classifier",
        "{var} = naive_bayes_classifier(df)", "naive_bayes_classifier",
    ),
    "knn_classifier": (
        "models.classification.basic", "knn_classifier",
        "{var} = knn_classifier(df)", "knn_classifier",
    ),
    # graph
    "graph_shortest_paths": (
        "models.graph.network", "graph_shortest_paths",
        "{var} = graph_shortest_paths(df)", "graph_shortest_paths",
    ),
    "graph_mst": (
        "models.graph.network", "graph_mst",
        "{var} = graph_mst(df)", "graph_mst",
    ),
    "graph_max_flow": (
        "models.graph.network", "graph_max_flow",
        "{var} = graph_max_flow(df)", "graph_max_flow",
    ),
    "graph_centrality": (
        "models.graph.network", "graph_centrality",
        "{var} = graph_centrality(df)", "graph_centrality",
    ),
    # queueing
    "queue_metrics": (
        "models.queueing.mmc", "queue_metrics",
        "{var} = queue_metrics(df)", "queue_metrics",
    ),
    # inventory
    "inventory_policy": (
        "models.inventory.policy", "inventory_policy",
        "{var} = inventory_policy(df)", "inventory_policy",
    ),
    # finance
    "var_cvar_risk": (
        "models.finance.risk", "var_cvar_risk",
        "{var} = var_cvar_risk(df)", "var_cvar_risk",
    ),
    "garch_volatility": (
        "models.finance.risk", "garch_volatility",
        "{var} = garch_volatility(df)", "garch_volatility",
    ),
    "black_scholes_pricing": (
        "models.finance.portfolio", "black_scholes_pricing",
        "{var} = black_scholes_pricing(df)", "black_scholes_pricing",
    ),
    "markowitz_portfolio": (
        "models.finance.portfolio", "markowitz_portfolio",
        "{var} = markowitz_portfolio(df)", "markowitz_portfolio",
    ),
    # association
    "apriori_rules": (
        "models.association.rules", "apriori_rules",
        "{var} = apriori_rules(df)", "apriori_rules",
    ),
    "granger_causality": (
        "models.association.rules", "granger_causality",
        "{var} = granger_causality(df)", "granger_causality",
    ),
    # signal
    "fft_frequency_analysis": (
        "models.signal.processing", "fft_frequency_analysis",
        "{var} = fft_frequency_analysis(df)", "fft_frequency_analysis",
    ),
    "signal_denoising": (
        "models.signal.processing", "signal_denoising",
        "{var} = signal_denoising(df)", "signal_denoising",
    ),
    "energy_detection": (
        "models.signal.processing", "energy_detection",
        "{var} = energy_detection(df)", "energy_detection",
    ),
}

# ── advanced models (use the generic ADVANCED_MODEL_BUILDERS for-loop) ──
# Each entry: (model_id, table_suffix, import_module, import_name, call_template)
# call_template can be a lambda string; None means the function ref itself is used directly.
ADVANCED_MODEL_REGISTRY: list[tuple[str, str, str, str, str | None]] = [
    # validation (always-on)
    ("error_analysis", "error_analysis", "models.validation.diagnostics", "error_analysis", None),
    ("sensitivity_analysis", "sensitivity_analysis", "models.validation.diagnostics", "sensitivity_analysis", None),
    ("model_comparison", "model_comparison", "models.validation.diagnostics", "model_comparison", None),
    # evaluation advanced
    ("ahp_entropy_combined", "ahp_entropy_combined", "models.evaluation.advanced", "ahp_entropy_combined_weights", None),
    ("dea_efficiency", "dea_efficiency", "models.evaluation.advanced", "dea_efficiency", None),
    ("fuzzy_evaluation", "fuzzy_evaluation", "models.evaluation.advanced", "fuzzy_comprehensive_evaluation", None),
    # optimization advanced
    ("nonlinear_optimization", "nonlinear_optimization", "models.optimization.advanced", "nonlinear_gradient_optimization", None),
    ("integer_programming", "integer_programming", "models.optimization.advanced", "integer_branch_bound", None),
    ("multiobjective_optimization", "multiobjective_optimization", "models.optimization.advanced", "multiobjective_weighted_sum", None),
    ("astar_path", "astar_path", "models.optimization.advanced", "astar_path_plan", None),
    ("tsp_route", "tsp_route", "models.optimization.advanced", "tsp_route_heuristic", None),
    ("vrp_route", "vrp_route", "models.optimization.advanced", "vrp_savings_heuristic", None),
    # prediction advanced
    ("seasonal_forecast", "seasonal_forecast", "models.prediction.advanced", "seasonal_decomposition_forecast", "frame, periods=3"),
    ("var_forecast", "var_forecast", "models.prediction.advanced", "var_forecast", "frame, periods=3"),
    ("nonlinear_forecast", "nonlinear_forecast", "models.prediction.advanced", "nonlinear_regression_forecast", "frame, periods=3"),
    # classification advanced
    ("smote_balance", "smote_balance", "models.classification.imbalance", "smote_balance_summary", None),
    # dimensionality advanced
    ("nonlinear_embedding", "nonlinear_embedding", "models.dimensionality.nonlinear", "nonlinear_embedding", None),
    # statistics advanced
    ("monte_carlo", "monte_carlo", "models.statistics.monte_carlo", "monte_carlo_simulation", None),
    # queueing advanced
    ("jackson_network", "jackson_network", "models.queueing.network", "jackson_network_queue", None),
    # inventory advanced
    ("multi_echelon_inventory", "multi_echelon_inventory", "models.inventory.supply_chain", "multi_echelon_inventory", None),
    ("bullwhip_effect", "bullwhip_effect", "models.inventory.supply_chain", "bullwhip_effect", None),
    # mechanism
    ("logistic_growth", "logistic_growth", "models.mechanism.dynamics", "logistic_growth_model", None),
    ("sir_model", "sir_model", "models.mechanism.dynamics", "sir_epidemic_model", None),
    ("lotka_volterra", "lotka_volterra", "models.mechanism.dynamics", "lotka_volterra_model", None),
    ("solow_growth", "solow_growth", "models.mechanism.dynamics", "solow_growth_model", None),
    ("heat_conduction", "heat_conduction", "models.mechanism.dynamics", "heat_conduction_1d", None),
    ("harmonic_oscillator", "harmonic_oscillator", "models.mechanism.dynamics", "harmonic_oscillator_model", None),
    ("michaelis_menten", "michaelis_menten", "models.mechanism.dynamics", "michaelis_menten_kinetics", None),
    ("bernoulli_flow", "bernoulli_flow", "models.mechanism.dynamics", "bernoulli_flow_analysis", None),
    # game
    ("nash_equilibrium", "nash_equilibrium", "models.game.theory", "nash_equilibrium_2x2", None),
    ("shapley_value", "shapley_value", "models.game.theory", "shapley_value", None),
    ("auction_pricing", "auction_pricing", "models.game.theory", "auction_pricing", None),
    ("stackelberg_equilibrium", "stackelberg_equilibrium", "models.game.theory", "stackelberg_equilibrium", None),
    # control
    ("kalman_filter", "kalman_filter", "models.control.estimation", "kalman_state_estimation", None),
    ("optimal_control", "optimal_control", "models.control.estimation", "optimal_control_dp", None),
    ("robust_control", "robust_control", "models.control.estimation", "robust_control_summary", None),
    # fitting advanced
    ("weighted_least_squares", "weighted_least_squares", "models.fitting.advanced", "weighted_least_squares_fit", None),
    ("nonlinear_fit", "nonlinear_fit", "models.fitting.advanced", "nonlinear_least_squares_fit", None),
    ("parameter_identification", "parameter_identification", "models.fitting.advanced", "parameter_identification", None),
    # image
    ("histogram_equalization", "histogram_equalization", "models.image.processing", "histogram_equalization", None),
    ("edge_detection", "edge_detection", "models.image.processing", "edge_detection_sobel", None),
    ("image_segmentation", "image_segmentation", "models.image.processing", "threshold_segmentation", None),
    ("image_features", "image_features", "models.image.processing", "image_feature_extraction", None),
    ("image_registration", "image_registration", "models.image.processing", "image_registration_shift", None),
    # transport
    ("traffic_flow", "traffic_flow", "models.transport.traffic", "traffic_flow_cellular", None),
    ("car_following", "car_following", "models.transport.traffic", "car_following_model", None),
    # ML prediction
    ("gradient_boosting", "gradient_boosting", "models.prediction.ml", "gradient_boosting_forecast", None),
    ("ridge_regression", "ridge_regression", "models.prediction.ml", "ridge_regression_forecast", None),
    # graph advanced
    ("community_detection", "community_detection", "models.graph.network", "community_detection", None),
    ("friend_recommendation", "friend_recommendation", "models.social_network.campus", "campus_friend_recommendation_model", None),
    ("information_propagation", "information_propagation", "models.social_network.campus", "campus_information_propagation_model", None),
    ("influence_maximization", "influence_maximization", "models.social_network.campus", "campus_influence_maximization_model", None),
]

# ── parameter sweep configurations ──
# model_id -> list of (param_name, default_value, sweep_values)
MODEL_SWEEP_CONFIGS: dict[str, list[tuple[str, float, list[float]]]] = {
    "sir_model": [
        ("beta", 0.3, [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("gamma", 0.1, [0.05, 0.1, 0.15, 0.2]),
    ],
    "grey_gm11": [("periods", 3, [2, 3, 4, 5, 6])],
    "smoothing_forecast": [("periods", 3, [2, 3, 4, 5, 6])],
    "trend_forecast": [("periods", 3, [2, 3, 4, 5, 6])],
    "seasonal_forecast": [("periods", 3, [2, 3, 4, 5, 6])],
    "kmeans_cluster": [("n_clusters", 3, [2, 3, 4, 5, 6])],
}


def basic_model_ids() -> set[str]:
    """Return model IDs handled by the direct dispatch code path."""
    return set(BASIC_MODEL_REGISTRY)


def advanced_model_ids() -> set[str]:
    """Return model IDs handled by the generic builder code path."""
    return {model_id for model_id, *_ in ADVANCED_MODEL_REGISTRY}


def registered_model_ids() -> set[str]:
    """Return every model ID that code generation can import and execute."""
    return basic_model_ids() | advanced_model_ids()


def registry_table_suffixes() -> set[str]:
    """Return every output table suffix declared by the registry."""
    suffixes = {entry[3] for entry in BASIC_MODEL_REGISTRY.values()}
    suffixes.update(suffix for _model_id, suffix, *_ in ADVANCED_MODEL_REGISTRY)
    return suffixes


def validate_model_registry(*, import_symbols: bool = False) -> list[str]:
    """Return configuration errors in the code-generation model registry.

    This is intentionally side-effect free unless ``import_symbols`` is true.
    Tests can opt into import validation so adding a model fails fast when the
    module path or function name is misspelled.
    """
    errors: list[str] = []
    basic_ids = basic_model_ids()
    advanced_ids = [model_id for model_id, *_ in ADVANCED_MODEL_REGISTRY]

    duplicate_advanced_ids = sorted(
        model_id for model_id, count in Counter(advanced_ids).items() if count > 1
    )
    for model_id in duplicate_advanced_ids:
        errors.append(f"duplicate advanced registry model_id: {model_id}")

    for model_id in sorted(basic_ids & set(advanced_ids)):
        errors.append(f"model_id registered as both basic and advanced: {model_id}")

    for model_id, entry in sorted(BASIC_MODEL_REGISTRY.items()):
        errors.extend(_validate_basic_entry(model_id, entry))

    for model_id, suffix, module_name, import_name, extra_args in ADVANCED_MODEL_REGISTRY:
        errors.extend(
            _validate_advanced_entry(model_id, suffix, module_name, import_name, extra_args)
        )

    registry_ids = registered_model_ids()
    for model_id in sorted(set(MODEL_SWEEP_CONFIGS) - registry_ids):
        errors.append(f"sweep config references unregistered model_id: {model_id}")

    if import_symbols:
        errors.extend(_validate_import_symbols(_iter_import_targets()))

    return errors


def _validate_basic_entry(model_id: str, entry: ModelEntry) -> list[str]:
    errors: list[str] = []
    module_name, import_name, call_template, table_suffix = entry
    if not _is_identifier_like(model_id):
        errors.append(f"invalid basic model_id: {model_id!r}")
    if not module_name:
        errors.append(f"{model_id}: missing import module")
    if not import_name:
        errors.append(f"{model_id}: missing import name")
    if "{var}" not in call_template:
        errors.append(f"{model_id}: basic call_template must include {{var}}")
    if not table_suffix:
        errors.append(f"{model_id}: missing table suffix")
    return errors


def _validate_advanced_entry(
    model_id: str,
    table_suffix: str,
    module_name: str,
    import_name: str,
    extra_args: str | None,
) -> list[str]:
    errors: list[str] = []
    if not _is_identifier_like(model_id):
        errors.append(f"invalid advanced model_id: {model_id!r}")
    if not table_suffix:
        errors.append(f"{model_id}: missing table suffix")
    if not module_name:
        errors.append(f"{model_id}: missing import module")
    if not import_name:
        errors.append(f"{model_id}: missing import name")
    if extra_args is not None and not extra_args.strip():
        errors.append(f"{model_id}: extra_args must be None or a non-empty string")
    return errors


def _validate_import_symbols(targets: Iterable[tuple[str, str, str]]) -> list[str]:
    errors: list[str] = []
    for model_id, module_name, import_name in targets:
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - exact import failures vary by env
            errors.append(f"{model_id}: cannot import module {module_name}: {exc}")
            continue
        if not hasattr(module, import_name):
            errors.append(f"{model_id}: {module_name} has no attribute {import_name}")
    return errors


def _iter_import_targets() -> Iterable[tuple[str, str, str]]:
    for model_id, (module_name, import_name, _call_template, _suffix) in BASIC_MODEL_REGISTRY.items():
        yield model_id, module_name, import_name
    for model_id, _suffix, module_name, import_name, _extra_args in ADVANCED_MODEL_REGISTRY:
        yield model_id, module_name, import_name


def _is_identifier_like(value: str) -> bool:
    return bool(value) and value[0].isalpha() and all(
        char.isalnum() or char == "_" for char in value
    )


def collect_imports(selected_model_ids: list[str]) -> list[str]:
    """Return deduplicated import lines for the given model IDs.

    Multiple symbols from the same module are combined into a single
    ``from module import name1, name2, ...`` line.
    """
    imports: dict[str, set[str]] = {}  # module -> {import_name, ...}
    for mid in selected_model_ids:
        entry = BASIC_MODEL_REGISTRY.get(mid)
        if entry:
            imports.setdefault(entry[0], set()).add(entry[1])
    for mid, _, mod, name, _ in ADVANCED_MODEL_REGISTRY:
        if mid in selected_model_ids:
            imports.setdefault(mod, set()).add(name)
    return [
        f"from {mod} import {', '.join(sorted(names))}"
        for mod, names in sorted(imports.items())
    ]
