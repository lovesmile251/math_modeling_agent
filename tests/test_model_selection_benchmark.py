from __future__ import annotations

import json

import pandas as pd
import pytest

from agents.base import WorkflowState
from agents.model_selection_agent import ModelSelectionAgent


def _run_selection(problem_text, workspace, data_files):
    state = WorkflowState(problem_text=problem_text, data_files=data_files, workspace=workspace)
    return ModelSelectionAgent().run(state)


def _benchmark_frame(profile: str) -> pd.DataFrame:
    if profile == "forecast":
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="ME").astype(str),
                "demand": [120, 132, 141, 150, 163, 171, 188, 197, 211, 226],
                "cost": [40, 42, 41, 45, 47, 48, 51, 53, 54, 56],
            }
        )
    if profile == "evaluation":
        return pd.DataFrame(
            {
                "object_id": ["A", "B", "C", "D", "E", "F"],
                "quality_score": [88, 76, 91, 83, 79, 95],
                "cost": [52, 47, 61, 48, 45, 66],
                "benefit": [120, 98, 141, 110, 104, 152],
            }
        )
    if profile == "capacity":
        return pd.DataFrame(
            {
                "region": ["north", "south", "east", "west"],
                "demand": [180, 220, 165, 205],
                "capacity": [160, 230, 155, 190],
                "cost": [12, 15, 11, 14],
            }
        )
    if profile == "optimization":
        return pd.DataFrame(
            {
                "project_id": [f"P{i}" for i in range(1, 7)],
                "resource_limit": [8, 6, 7, 5, 9, 4],
                "cost": [42, 35, 38, 26, 49, 22],
                "profit": [78, 66, 71, 50, 88, 43],
                "capacity": [10, 8, 9, 7, 11, 6],
            }
        )
    if profile == "classification":
        return pd.DataFrame(
            {
                "feature_a": [1.0, 1.3, 1.7, 2.2, 2.8, 3.1, 3.6, 4.0, 4.4, 4.8],
                "feature_b": [5.2, 5.0, 4.6, 4.1, 3.8, 3.4, 2.8, 2.4, 2.1, 1.7],
                "label": ["A", "A", "A", "A", "B", "B", "B", "B", "B", "A"],
            }
        )
    if profile == "cluster":
        return pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(1, 9)],
                "spend": [12, 14, 13, 45, 48, 46, 82, 85],
                "frequency": [2, 3, 2, 8, 9, 8, 13, 14],
                "recency": [30, 28, 35, 12, 10, 14, 3, 4],
            }
        )
    if profile == "network":
        return pd.DataFrame(
            {
                "source": ["A", "A", "B", "C", "D", "E"],
                "target": ["B", "C", "D", "D", "E", "F"],
                "distance": [4, 2, 5, 1, 3, 6],
                "capacity": [20, 15, 18, 12, 10, 9],
            }
        )
    if profile == "statistics":
        return pd.DataFrame(
            {
                "sample_id": [f"S{i}" for i in range(1, 9)],
                "input_x": [1, 2, 3, 4, 5, 6, 7, 8],
                "input_z": [2, 1, 4, 3, 6, 5, 8, 7],
                "target": [3, 5, 7, 9, 11, 13, 15, 17],
                "group": ["control", "control", "control", "test", "test", "test", "test", "control"],
            }
        )
    if profile == "simulation":
        return pd.DataFrame(
            {
                "trial": list(range(1, 9)),
                "arrival_rate": [3.1, 3.4, 3.0, 3.8, 4.1, 4.0, 4.4, 4.6],
                "service_rate": [5.2, 5.0, 5.1, 5.4, 5.6, 5.3, 5.8, 6.0],
                "risk_probability": [0.12, 0.14, 0.10, 0.18, 0.20, 0.17, 0.22, 0.24],
            }
        )
    raise AssertionError(f"unknown benchmark profile: {profile}")


MODEL_SELECTION_BENCHMARK_CASES = [
    {
        "case": "trend forecast demand",
        "problem": "Forecast future demand with a time series trend and compare prediction error.",
        "profile": "forecast",
        "expected_primary": {"trend_forecast"},
        "expected_task_types": {"forecast"},
    },
    {
        "case": "small sample grey forecast",
        "problem": "Use grey forecast GM(1,1) for a small sample future demand forecast.",
        "profile": "forecast",
        "expected_primary": {"grey_gm11", "trend_forecast"},
        "expected_task_types": {"forecast"},
    },
    {
        "case": "seasonal forecast",
        "problem": "Build a seasonal time series forecast for monthly demand.",
        "profile": "forecast",
        "expected_primary": {"seasonal_forecast", "trend_forecast"},
        "expected_task_types": {"forecast"},
    },
    {
        "case": "multi indicator evaluation",
        "problem": "Rank alternatives by comprehensive evaluation, objective weights, and score indicators.",
        "profile": "evaluation",
        "expected_primary": {"entropy_weights", "topsis_rank"},
        "expected_task_types": {"evaluation"},
    },
    {
        "case": "ahp evaluation",
        "problem": "Use AHP hierarchy analysis with expert weights for supplier evaluation and ranking.",
        "profile": "evaluation",
        "expected_primary": {"ahp_weights", "entropy_weights", "topsis_rank"},
        "expected_task_types": {"evaluation"},
    },
    {
        "case": "capacity gap",
        "problem": "Analyze demand and capacity gap across regions and identify shortages.",
        "profile": "capacity",
        "expected_primary": {"capacity_gap"},
        "expected_task_types": {"evaluation", "optimization"},
    },
    {
        "case": "resource allocation",
        "problem": "Optimize resource allocation under budget, cost, profit, and capacity constraints.",
        "profile": "optimization",
        "expected_primary": {"resource_allocation"},
        "expected_task_types": {"optimization"},
    },
    {
        "case": "knapsack selection",
        "problem": "Solve a 0-1 knapsack project selection problem under budget capacity constraints.",
        "profile": "optimization",
        "expected_primary": {"knapsack_01", "resource_allocation"},
        "expected_task_types": {"optimization"},
    },
    {
        "case": "assignment plan",
        "problem": "Optimize assignment and matching of workers to tasks while minimizing total cost.",
        "profile": "optimization",
        "expected_primary": {"assignment_plan", "resource_allocation"},
        "expected_task_types": {"optimization"},
    },
    {
        "case": "scheduling plan",
        "problem": "Schedule jobs on machines with deadlines, resource limits, and cost constraints.",
        "profile": "optimization",
        "expected_primary": {"scheduling_plan", "resource_allocation"},
        "expected_task_types": {"optimization"},
    },
    {
        "case": "binary classification",
        "problem": "Train a classifier to identify class labels from numeric features.",
        "profile": "classification",
        "expected_primary": {"logistic_classifier", "naive_bayes_classifier", "knn_classifier"},
        "expected_task_types": {"classification"},
    },
    {
        "case": "customer clustering",
        "problem": "Cluster unlabeled customers into segments from behavior features.",
        "profile": "cluster",
        "expected_primary": {"kmeans_cluster", "dbscan_cluster", "hierarchical_cluster"},
        "expected_task_types": {"clustering"},
    },
    {
        "case": "pca dimensionality",
        "problem": "Use PCA dimensionality reduction before clustering customer segments.",
        "profile": "cluster",
        "expected_primary": {"pca", "kmeans_cluster"},
        "expected_task_types": {"clustering", "statistics"},
    },
    {
        "case": "shortest path network",
        "problem": "Find shortest path in a network using source, target, and edge distance.",
        "profile": "network",
        "expected_primary": {"graph_shortest_paths"},
        "expected_task_types": {"network"},
    },
    {
        "case": "network flow",
        "problem": "Compute maximum flow on a directed network with source target edge capacity.",
        "profile": "network",
        "expected_primary": {"graph_max_flow", "graph_centrality"},
        "expected_task_types": {"network"},
    },
    {
        "case": "community network",
        "problem": "Detect network communities and rank node centrality in a graph.",
        "profile": "network",
        "expected_primary": {"community_detection", "graph_centrality"},
        "expected_task_types": {"network", "clustering"},
    },
    {
        "case": "correlation regression statistics",
        "problem": "Run correlation analysis and linear regression to explain target relationships.",
        "profile": "statistics",
        "expected_primary": {"correlation_analysis", "linear_regression"},
        "expected_task_types": {"statistics"},
    },
    {
        "case": "hypothesis test statistics",
        "problem": "Perform statistical correlation checks and hypothesis tests to compare control and test groups.",
        "profile": "statistics",
        "expected_primary": {"hypothesis_tests", "parameter_estimation"},
        "expected_task_types": {"statistics"},
    },
    {
        "case": "monte carlo simulation",
        "problem": "Use Monte Carlo simulation to estimate uncertainty and risk probability.",
        "profile": "simulation",
        "expected_primary": {"monte_carlo"},
        "expected_task_types": {"simulation"},
    },
    {
        "case": "mixed modeling subquestions",
        "problem": (
            "Q1 rank regions by comprehensive evaluation score. "
            "Q2 forecast future demand trend. "
            "Q3 optimize resource allocation under capacity and cost constraints."
        ),
        "profile": "optimization",
        "expected_primary": {"entropy_weights", "topsis_rank", "trend_forecast", "resource_allocation"},
        "expected_task_types": {"evaluation", "forecast", "optimization"},
    },
]


@pytest.mark.parametrize("case", MODEL_SELECTION_BENCHMARK_CASES, ids=lambda item: item["case"])
def test_model_selection_benchmark_covers_expected_primary_or_task_type(temp_workspace, case):
    path = temp_workspace.data_dir / f"{case['case'].replace(' ', '_')}.csv"
    _benchmark_frame(case["profile"]).to_csv(path, index=False)

    state = _run_selection(case["problem"], temp_workspace, [path])
    selected = set(json.loads(state.notes["selected_model_ids"]))
    payload = json.loads(state.artifacts["model_selection_report"].read_text(encoding="utf-8"))
    task_types = {task["task_type"] for task in payload["tasks"]}
    selected_task_types = {model["task_type"] for model in payload["selected_models"]}
    covered_task_types = task_types | selected_task_types

    primary_model = next(iter(json.loads(state.notes["selected_model_ids"])), "")
    primary_hit = primary_model in case["expected_primary"]
    task_type_hit = case["expected_task_types"].issubset(covered_task_types)

    assert primary_hit, (
        f"case={case['case']!r} missed expected coverage\n"
        f"expected_primary={sorted(case['expected_primary'])}\n"
        f"expected_task_types={sorted(case['expected_task_types'])}\n"
        f"primary_model={primary_model!r}\n"
        f"selected={sorted(selected)}\n"
        f"tasks={sorted(task_types)}\n"
        f"selected_task_types={sorted(selected_task_types)}"
    )
    assert task_type_hit
