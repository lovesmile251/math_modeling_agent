from __future__ import annotations

import pandas as pd
import pytest

from models.catalog import EXECUTABLE_MODEL_LABELS


# ---------------------------------------------------------------------------
# model_id → (function, needs_periods, needs_window, needs_target)
# Derived from agents/coding_agent.py dispatch logic.
# ---------------------------------------------------------------------------

def _build_model_mapping():
    """Lazy-import each model function so missing deps don't break the whole suite."""
    mapping = {}

    # -- evaluation ---------------------------------------------------------
    from models.evaluation.entropy_weight import entropy_weights
    mapping["entropy_weights"] = (entropy_weights, False, False, False)

    from models.evaluation.topsis import topsis_rank
    mapping["topsis_rank"] = (topsis_rank, False, False, False)

    from models.evaluation.ahp import ahp_weights
    mapping["ahp_weights"] = (ahp_weights, False, False, False)

    from models.evaluation.grey_relation import grey_relation_rank
    mapping["grey_relation"] = (grey_relation_rank, False, False, False)

    from models.evaluation.vikor import vikor_rank
    mapping["vikor"] = (vikor_rank, False, False, False)

    from models.evaluation.advanced import ahp_entropy_combined_weights, dea_efficiency, fuzzy_comprehensive_evaluation
    mapping["ahp_entropy_combined"] = (ahp_entropy_combined_weights, False, False, False)
    mapping["dea_efficiency"] = (dea_efficiency, False, False, False)
    mapping["fuzzy_evaluation"] = (fuzzy_comprehensive_evaluation, False, False, False)

    # -- diagnostics --------------------------------------------------------
    from models.diagnostics.capacity import demand_capacity_gap
    mapping["capacity_gap"] = (demand_capacity_gap, False, False, False)

    # -- optimization -------------------------------------------------------
    from models.optimization.resource_allocation import resource_allocation_plan
    mapping["resource_allocation"] = (resource_allocation_plan, False, False, False)

    from models.optimization.discrete import assignment_plan, bin_packing_plan, knapsack_01_plan, scheduling_plan
    mapping["knapsack_01"] = (knapsack_01_plan, False, False, False)
    mapping["assignment_plan"] = (assignment_plan, False, False, False)
    mapping["bin_packing"] = (bin_packing_plan, False, False, False)
    mapping["scheduling_plan"] = (scheduling_plan, False, False, False)

    from models.optimization.esp import cement_esp_optimization
    mapping["cement_esp_optimization"] = (cement_esp_optimization, False, False, False)

    from models.optimization.planting import crop_planting_plan
    mapping["crop_planting_plan"] = (crop_planting_plan, False, False, False)

    from models.optimization.advanced import (
        astar_path_plan,
        integer_branch_bound,
        multiobjective_weighted_sum,
        nonlinear_gradient_optimization,
        tsp_route_heuristic,
        vrp_savings_heuristic,
    )
    mapping["nonlinear_optimization"] = (nonlinear_gradient_optimization, False, False, False)
    mapping["integer_programming"] = (integer_branch_bound, False, False, False)
    mapping["multiobjective_optimization"] = (multiobjective_weighted_sum, False, False, False)
    mapping["astar_path"] = (astar_path_plan, False, False, False)
    mapping["tsp_route"] = (tsp_route_heuristic, False, False, False)
    mapping["vrp_route"] = (vrp_savings_heuristic, False, False, False)

    # -- prediction ---------------------------------------------------------
    from models.prediction.trend import linear_trend_forecast
    mapping["trend_forecast"] = (linear_trend_forecast, True, False, False)

    from models.prediction.grey_gm import grey_gm11_forecast
    mapping["grey_gm11"] = (grey_gm11_forecast, True, False, False)

    from models.prediction.smoothing import smoothing_forecast
    mapping["smoothing_forecast"] = (smoothing_forecast, True, False, False)

    from models.prediction.advanced import nonlinear_regression_forecast, seasonal_decomposition_forecast, var_forecast
    mapping["seasonal_forecast"] = (seasonal_decomposition_forecast, True, False, False)
    mapping["var_forecast"] = (var_forecast, True, False, False)
    mapping["nonlinear_forecast"] = (nonlinear_regression_forecast, True, False, False)

    from models.prediction.ml import gradient_boosting_forecast, ridge_regression_forecast
    mapping["gradient_boosting"] = (gradient_boosting_forecast, False, False, False)
    mapping["ridge_regression"] = (ridge_regression_forecast, False, False, False)

    # -- classification -----------------------------------------------------
    from models.classification.basic import knn_classifier, logistic_binary_classifier, naive_bayes_classifier
    mapping["logistic_classifier"] = (logistic_binary_classifier, False, False, False)
    mapping["naive_bayes_classifier"] = (naive_bayes_classifier, False, False, False)
    mapping["knn_classifier"] = (knn_classifier, False, False, False)

    from models.classification.imbalance import smote_balance_summary
    mapping["smote_balance"] = (smote_balance_summary, False, False, False)

    # -- clustering ---------------------------------------------------------
    from models.clustering.kmeans import kmeans_cluster
    mapping["kmeans_cluster"] = (kmeans_cluster, False, False, False)

    from models.clustering.density import dbscan_cluster
    mapping["dbscan_cluster"] = (dbscan_cluster, False, False, False)

    from models.clustering.hierarchical import hierarchical_cluster
    mapping["hierarchical_cluster"] = (hierarchical_cluster, False, False, False)

    # -- dimensionality -----------------------------------------------------
    from models.dimensionality.pca import pca_summary
    mapping["pca"] = (pca_summary, False, False, False)

    from models.dimensionality.feature_selection import feature_selection
    mapping["feature_selection"] = (feature_selection, False, False, False)

    from models.dimensionality.nonlinear import nonlinear_embedding
    mapping["nonlinear_embedding"] = (nonlinear_embedding, False, False, False)

    # -- statistics ---------------------------------------------------------
    from models.statistics.correlation import correlation_analysis
    mapping["correlation_analysis"] = (correlation_analysis, False, False, False)

    from models.statistics.regression import linear_regression_summary
    mapping["linear_regression"] = (linear_regression_summary, False, False, False)

    from models.statistics.estimation import parameter_estimation
    mapping["parameter_estimation"] = (parameter_estimation, False, False, False)

    from models.statistics.hypothesis import hypothesis_tests
    mapping["hypothesis_tests"] = (hypothesis_tests, False, False, False)

    from models.statistics.anova import anova_analysis
    mapping["anova_analysis"] = (anova_analysis, False, False, False)

    from models.statistics.sampling import quality_sampling_plan
    mapping["quality_sampling_plan"] = (quality_sampling_plan, False, False, False)

    from models.statistics.nipt import nipt_bmi_grouping
    mapping["nipt_bmi_grouping"] = (nipt_bmi_grouping, False, False, False)

    from models.statistics.monte_carlo import monte_carlo_simulation
    mapping["monte_carlo"] = (monte_carlo_simulation, False, False, False)

    # -- graph / network ----------------------------------------------------
    from models.graph.network import community_detection, graph_centrality, graph_max_flow, graph_mst, graph_shortest_paths
    mapping["graph_shortest_paths"] = (graph_shortest_paths, False, False, False)
    mapping["graph_mst"] = (graph_mst, False, False, False)
    mapping["graph_max_flow"] = (graph_max_flow, False, False, False)
    mapping["graph_centrality"] = (graph_centrality, False, False, False)
    mapping["community_detection"] = (community_detection, False, False, False)

    from models.social_network.campus import (
        campus_friend_recommendation_model,
        campus_influence_maximization_model,
        campus_information_propagation_model,
    )
    mapping["friend_recommendation"] = (campus_friend_recommendation_model, False, False, False)
    mapping["information_propagation"] = (campus_information_propagation_model, False, False, False)
    mapping["influence_maximization"] = (campus_influence_maximization_model, False, False, False)

    # -- queueing -----------------------------------------------------------
    from models.queueing.mmc import queue_metrics
    mapping["queue_metrics"] = (queue_metrics, False, False, False)

    from models.queueing.network import jackson_network_queue
    mapping["jackson_network"] = (jackson_network_queue, False, False, False)

    # -- inventory ----------------------------------------------------------
    from models.inventory.policy import inventory_policy
    mapping["inventory_policy"] = (inventory_policy, False, False, False)

    from models.inventory.supply_chain import bullwhip_effect, multi_echelon_inventory
    mapping["multi_echelon_inventory"] = (multi_echelon_inventory, False, False, False)
    mapping["bullwhip_effect"] = (bullwhip_effect, False, False, False)

    # -- finance ------------------------------------------------------------
    from models.finance.risk import garch_volatility, var_cvar_risk
    mapping["var_cvar_risk"] = (var_cvar_risk, False, False, False)
    mapping["garch_volatility"] = (garch_volatility, False, False, False)

    from models.finance.portfolio import black_scholes_pricing, markowitz_portfolio
    mapping["black_scholes_pricing"] = (black_scholes_pricing, False, False, False)
    mapping["markowitz_portfolio"] = (markowitz_portfolio, False, False, False)

    # -- association --------------------------------------------------------
    from models.association.rules import apriori_rules, granger_causality
    mapping["apriori_rules"] = (apriori_rules, False, False, False)
    mapping["granger_causality"] = (granger_causality, False, False, False)

    # -- signal -------------------------------------------------------------
    from models.signal.processing import energy_detection, fft_frequency_analysis, signal_denoising
    mapping["fft_frequency_analysis"] = (fft_frequency_analysis, False, False, False)
    mapping["signal_denoising"] = (signal_denoising, False, True, False)
    mapping["energy_detection"] = (energy_detection, False, False, False)

    # -- mechanism ----------------------------------------------------------
    from models.mechanism.dynamics import (
        bernoulli_flow_analysis,
        harmonic_oscillator_model,
        heat_conduction_1d,
        logistic_growth_model,
        lotka_volterra_model,
        michaelis_menten_kinetics,
        sir_epidemic_model,
        solow_growth_model,
    )
    mapping["logistic_growth"] = (logistic_growth_model, False, False, False)
    mapping["sir_model"] = (sir_epidemic_model, False, False, False)
    mapping["lotka_volterra"] = (lotka_volterra_model, False, False, False)
    mapping["solow_growth"] = (solow_growth_model, False, False, False)
    mapping["heat_conduction"] = (heat_conduction_1d, False, False, False)
    mapping["harmonic_oscillator"] = (harmonic_oscillator_model, False, False, False)
    mapping["michaelis_menten"] = (michaelis_menten_kinetics, False, False, False)
    mapping["bernoulli_flow"] = (bernoulli_flow_analysis, False, False, False)

    # -- game theory --------------------------------------------------------
    from models.game.theory import auction_pricing, nash_equilibrium_2x2, shapley_value, stackelberg_equilibrium
    mapping["nash_equilibrium"] = (nash_equilibrium_2x2, False, False, False)
    mapping["shapley_value"] = (shapley_value, False, False, False)
    mapping["auction_pricing"] = (auction_pricing, False, False, False)
    mapping["stackelberg_equilibrium"] = (stackelberg_equilibrium, False, False, False)

    # -- control ------------------------------------------------------------
    from models.control.estimation import kalman_state_estimation, optimal_control_dp, robust_control_summary
    mapping["kalman_filter"] = (kalman_state_estimation, False, False, False)
    mapping["optimal_control"] = (optimal_control_dp, False, False, False)
    mapping["robust_control"] = (robust_control_summary, False, False, False)

    # -- fitting ------------------------------------------------------------
    from models.fitting.polynomial import polynomial_fit
    mapping["polynomial_fit"] = (polynomial_fit, False, False, False)

    from models.fitting.advanced import nonlinear_least_squares_fit, parameter_identification, weighted_least_squares_fit
    mapping["weighted_least_squares"] = (weighted_least_squares_fit, False, False, False)
    mapping["nonlinear_fit"] = (nonlinear_least_squares_fit, False, False, False)
    mapping["parameter_identification"] = (parameter_identification, False, False, False)

    # -- image --------------------------------------------------------------
    from models.image.processing import (
        edge_detection_sobel,
        histogram_equalization,
        image_feature_extraction,
        image_registration_shift,
        threshold_segmentation,
    )
    mapping["histogram_equalization"] = (histogram_equalization, False, False, False)
    mapping["edge_detection"] = (edge_detection_sobel, False, False, False)
    mapping["image_segmentation"] = (threshold_segmentation, False, False, False)
    mapping["image_features"] = (image_feature_extraction, False, False, False)
    mapping["image_registration"] = (image_registration_shift, False, False, False)

    # -- transport ----------------------------------------------------------
    from models.transport.traffic import car_following_model, traffic_flow_cellular
    mapping["traffic_flow"] = (traffic_flow_cellular, False, False, False)
    mapping["car_following"] = (car_following_model, False, False, False)

    # -- validation ---------------------------------------------------------
    from models.validation.diagnostics import error_analysis, model_comparison, sensitivity_analysis
    mapping["error_analysis"] = (error_analysis, False, False, False)
    mapping["sensitivity_analysis"] = (sensitivity_analysis, False, False, False)
    mapping["model_comparison"] = (model_comparison, False, False, False)

    return mapping


MODEL_FUNCTION_MAP = _build_model_mapping()


# ---------------------------------------------------------------------------
# Helper to create a plausible DataFrame for any model
# ---------------------------------------------------------------------------

def _make_test_df(model_id: str) -> pd.DataFrame:
    """Return a small numeric DataFrame that most models can consume."""
    import numpy as np
    rng = np.random.default_rng(42)

    # Models that need specific column names
    if model_id == "capacity_gap":
        return pd.DataFrame({
            "demand": rng.integers(80, 200, 10).astype(float),
            "capacity": rng.integers(100, 250, 10).astype(float),
            "cost": rng.integers(30, 60, 10).astype(float),
        })

    if model_id in ("resource_allocation", "knapsack_01", "assignment_plan",
                    "bin_packing", "scheduling_plan",
                    "nonlinear_optimization", "integer_programming",
                    "multiobjective_optimization"):
        return pd.DataFrame({
            "cost": rng.integers(10, 100, 10).astype(float),
            "benefit": rng.integers(20, 200, 10).astype(float),
            "resource": rng.integers(5, 50, 10).astype(float),
            "budget": rng.integers(100, 500, 10).astype(float),
            "profit": rng.integers(30, 150, 10).astype(float),
        })

    if model_id == "cement_esp_optimization":
        rows = 48
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "Temp_C": rng.normal(126, 4, rows),
            "C_in_gNm3": rng.normal(36, 5, rows),
            "Q_Nm3h": rng.normal(462000, 12000, rows),
            "U1_kV": rng.normal(59, 3, rows),
            "U2_kV": rng.normal(59, 3, rows),
            "U3_kV": rng.normal(49, 2, rows),
            "U4_kV": rng.normal(49, 2, rows),
            "T1_s": rng.normal(230, 10, rows),
            "T2_s": rng.normal(230, 10, rows),
            "T3_s": rng.normal(440, 12, rows),
            "T4_s": rng.normal(440, 12, rows),
            "C_out_mgNm3": rng.normal(49.8, 0.2, rows),
            "P_total_kW": rng.normal(1770, 70, rows),
        })

    if model_id == "crop_planting_plan":
        return pd.DataFrame({
            "crop": ["wheat", "corn", "soybean", "rice"],
            "plot": ["A", "B", "C", "D"],
            "area": [30, 25, 20, 18],
            "yield_per_area": [1.2, 1.5, 0.9, 1.8],
            "price": [2.4, 2.1, 3.0, 2.8],
            "cost": [1.1, 1.3, 0.9, 1.7],
            "demand": [40, 35, 18, 32],
        })

    if model_id in ("var_cvar_risk", "garch_volatility", "black_scholes_pricing",
                    "markowitz_portfolio"):
        return pd.DataFrame({
            "return": rng.normal(0.001, 0.02, 10),
            "price": rng.uniform(50, 200, 10),
            "asset_a": rng.normal(0.01, 0.03, 10),
            "asset_b": rng.normal(0.008, 0.025, 10),
            "asset_c": rng.normal(0.012, 0.028, 10),
        })

    if model_id in ("graph_shortest_paths", "graph_mst", "graph_max_flow",
                    "graph_centrality", "community_detection",
                    "astar_path", "tsp_route", "vrp_route"):
        return pd.DataFrame({
            "source": [f"node{i}" for i in range(10)],
            "target": [f"node{(i + 3) % 10}" for i in range(10)],
            "weight": rng.uniform(1, 20, 10),
            "capacity": rng.integers(10, 50, 10).astype(float),
            "distance": rng.uniform(5, 100, 10),
        })

    if model_id in ("friend_recommendation", "information_propagation",
                    "influence_maximization"):
        return pd.DataFrame({
            "source": [
                "u0", "u0", "u0", "u1", "u1", "u2", "u2", "u3",
                "u4", "u4", "u5", "u6", "u7", "u8", "u9",
            ],
            "target": [
                "u1", "u2", "u3", "u2", "u4", "u5", "u6", "u7",
                "u5", "u8", "u9", "u9", "u8", "u9", "u10",
            ],
            "weight": rng.uniform(1, 5, 15),
        })

    if model_id in ("inventory_policy", "multi_echelon_inventory", "bullwhip_effect"):
        return pd.DataFrame({
            "demand": rng.integers(50, 200, 10).astype(float),
            "capacity": rng.integers(100, 300, 10).astype(float),
            "cost": rng.integers(10, 50, 10).astype(float),
            "lead_time": rng.integers(1, 7, 10).astype(float),
            "inventory": rng.integers(20, 150, 10).astype(float),
        })

    if model_id in ("queue_metrics", "jackson_network"):
        return pd.DataFrame({
            "arrival_rate": rng.uniform(1, 10, 10),
            "service_rate": rng.uniform(5, 20, 10),
            "servers": rng.integers(1, 5, 10).astype(float),
            "waiting": rng.integers(0, 20, 10).astype(float),
            "service_time": rng.uniform(0.5, 5, 10),
        })

    if model_id in ("logistic_growth", "sir_model", "lotka_volterra",
                    "solow_growth", "heat_conduction", "harmonic_oscillator",
                    "michaelis_menten", "bernoulli_flow"):
        return pd.DataFrame({
            "time": rng.uniform(0, 10, 10),
            "population": rng.integers(100, 1000, 10).astype(float),
            "rate": rng.uniform(0.01, 0.5, 10),
            "volume": rng.uniform(1, 100, 10),
            "pressure": rng.uniform(10, 200, 10),
        })

    if model_id in ("nash_equilibrium", "stackelberg_equilibrium",
                    "shapley_value", "auction_pricing"):
        return pd.DataFrame({
            "player_a_payoff": rng.integers(1, 10, 10).astype(float),
            "player_b_payoff": rng.integers(1, 10, 10).astype(float),
            "strategy": [f"s{i}" for i in range(10)],
            "value": rng.uniform(5, 100, 10),
            "bid": rng.uniform(1, 50, 10),
        })

    if model_id in ("kalman_filter", "optimal_control", "robust_control"):
        return pd.DataFrame({
            "measurement": rng.normal(10, 2, 10),
            "state": rng.normal(5, 1, 10),
            "control": rng.uniform(-2, 2, 10),
            "noise": rng.normal(0, 0.5, 10),
            "time": rng.uniform(0, 10, 10),
        })

    if model_id in ("histogram_equalization", "edge_detection",
                    "image_segmentation", "image_features", "image_registration"):
        # 8x8 image-like data
        rows = []
        for i in range(8):
            for j in range(8):
                rows.append({"row": i, "col": j, "pixel": float(rng.integers(0, 256))})
        return pd.DataFrame(rows)

    if model_id in ("apriori_rules", "granger_causality"):
        return pd.DataFrame({
            "item_a": rng.integers(0, 2, 10).astype(float),
            "item_b": rng.integers(0, 2, 10).astype(float),
            "item_c": rng.integers(0, 2, 10).astype(float),
            "value": rng.normal(0, 1, 10),
            "weight": rng.uniform(0.5, 1.5, 10),
        })

    if model_id == "quality_sampling_plan":
        return pd.DataFrame({
            "sample_count": [100, 120, 80],
            "defect_count": [8, 11, 7],
            "defect_rate": [0.08, 0.0917, 0.0875],
        })

    if model_id == "nipt_bmi_grouping":
        return pd.DataFrame({
            "bmi": [23.5, 25.0, 27.8, 29.2, 30.5, 32.1, 34.0, 35.5],
            "gestational_week": [11, 12, 12, 13, 13, 14, 15, 15],
            "y_concentration": [0.032, 0.041, 0.044, 0.038, 0.046, 0.039, 0.043, 0.050],
            "abnormal": [0, 0, 0, 0, 1, 0, 1, 0],
        })

    if model_id in ("fft_frequency_analysis", "signal_denoising", "energy_detection"):
        t = rng.uniform(0, 1, 50)
        return pd.DataFrame({
            "time": t,
            "signal": 2.0 * __import__("numpy").sin(2 * __import__("numpy").pi * 5 * t) + rng.normal(0, 0.3, 50),
            "amplitude": rng.uniform(0.5, 1.5, 50),
            "frequency": rng.uniform(1, 10, 50),
        })

    if model_id in ("traffic_flow", "car_following"):
        return pd.DataFrame({
            "position": rng.uniform(0, 100, 10),
            "speed": rng.uniform(5, 30, 10),
            "density": rng.uniform(5, 50, 10),
            "flow": rng.integers(100, 500, 10).astype(float),
            "headway": rng.uniform(1, 10, 10),
        })

    # Default: generic numeric DataFrame
    return pd.DataFrame({
        "col_a": rng.normal(10, 3, 10),
        "col_b": rng.normal(20, 5, 10),
        "col_c": rng.normal(30, 7, 10),
        "col_d": rng.integers(0, 100, 10).astype(float),
        "col_e": rng.uniform(1, 50, 10),
        "label": rng.integers(0, 2, 10).astype(float),
    })


# ---------------------------------------------------------------------------
# Parameterized smoke test
# ---------------------------------------------------------------------------

def _model_ids():
    """Return every model_id that has a registered function mapping."""
    missing = sorted(set(EXECUTABLE_MODEL_LABELS) - set(MODEL_FUNCTION_MAP))
    if missing:
        pytest.fail(f"EXECUTABLE_MODEL_LABELS contains models without a function mapping: {missing}")
    return list(EXECUTABLE_MODEL_LABELS)


@pytest.mark.parametrize("model_id", _model_ids())
def test_model_smoke(model_id: str):
    """Every executable model must accept a DataFrame and return a DataFrame (or Series)."""
    func, needs_periods, needs_window, needs_target = MODEL_FUNCTION_MAP[model_id]
    df = _make_test_df(model_id)

    # Build kwargs
    kwargs = {}
    if needs_periods:
        kwargs["periods"] = 3
    if needs_window:
        kwargs["window"] = 3

    # Call the model
    result = func(df, **kwargs)

    # Verify result type
    if isinstance(result, pd.Series):
        # entropy_weights returns a Series — that's acceptable
        assert result is not None, f"{model_id} returned None Series"
    else:
        assert isinstance(result, pd.DataFrame), f"{model_id} returned {type(result).__name__}, expected DataFrame or Series"
        assert result is not None, f"{model_id} returned None"
