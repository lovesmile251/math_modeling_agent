from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.classification.basic import logistic_binary_classifier
from models.diagnostics.capacity import demand_capacity_gap
from models.evaluation.entropy_weight import entropy_weights
from models.evaluation.topsis import topsis_rank
from models.inventory.policy import inventory_policy
from models.prediction.ml import gradient_boosting_forecast
from models.prediction.trend import linear_trend_forecast
from models.statistics.correlation import correlation_analysis


def test_trend_forecast_returns_predictions(sample_dataframe):
    result = linear_trend_forecast(sample_dataframe, periods=3)
    assert not result.empty
    assert {"target", "forecast_step", "forecast", "slope", "r_squared"}.issubset(result.columns)
    # 3 non-time targets x 3 periods = 9 rows (year inferred as time column)
    assert result["forecast_step"].max() == 3


def test_trend_forecast_empty_on_insufficient_rows():
    tiny = pd.DataFrame({"value": [1.0]})
    assert linear_trend_forecast(tiny).empty


def test_trend_forecast_empty_on_no_numeric():
    text_only = pd.DataFrame({"name": ["a", "b", "c"]})
    assert linear_trend_forecast(text_only).empty


def test_entropy_weights_sum_to_one(sample_dataframe):
    weights = entropy_weights(sample_dataframe)
    assert not weights.empty
    assert pytest.approx(float(weights.sum()), abs=1e-6) == 1.0
    assert (weights >= 0).all()


def test_entropy_weights_empty_on_no_numeric():
    text_only = pd.DataFrame({"name": ["a", "b"]})
    assert entropy_weights(text_only).empty


def test_topsis_rank_orders_alternatives(sample_dataframe):
    result = topsis_rank(sample_dataframe)
    assert not result.empty
    # every alternative gets a unique rank covering 1..n
    ranks = sorted(result["rank"].tolist())
    assert ranks == list(range(1, len(ranks) + 1))


def test_capacity_gap_detects_supply_demand(sample_dataframe):
    result = demand_capacity_gap(sample_dataframe)
    assert not result.empty
    assert "gap" in result.columns
    # capacity >= demand for every row in the sample, so gaps are non-negative
    assert (result["gap"] >= 0).all()


def test_capacity_gap_empty_without_relevant_columns():
    df = pd.DataFrame({"alpha": [1, 2, 3], "beta": [4, 5, 6]})
    assert demand_capacity_gap(df).empty


def test_inventory_policy_handles_sales_price_and_loss_rate_tables():
    sales = pd.DataFrame(
        {
            "单品编码": [101, 101, 102],
            "销量(千克)": [12.0, 18.0, 9.0],
            "批发价格(元/千克)": [4.0, 4.2, 6.0],
        }
    )
    price_only = pd.DataFrame(
        {
            "单品编码": [101, 102, 103],
            "批发价格(元/千克)": [4.0, 6.0, 8.0],
        }
    )
    loss_only = pd.DataFrame(
        {
            "小分类名称": ["花叶类", "食用菌"],
            "平均损耗率(%)": [12.8, 9.4],
        }
    )

    for frame in (sales, price_only, loss_only):
        result = inventory_policy(frame)
        assert not result.empty
        assert {
            "item_key",
            "suggested_replenishment",
            "price_markup_floor",
            "method",
        }.issubset(result.columns)
        assert (result["suggested_replenishment"] > 0).all()


def test_gradient_boosting_forecast_bounds_wide_table_cost():
    rng = np.random.default_rng(7)
    feature_data = {f"feature_{index}": rng.normal(index, 1.0, 250) for index in range(40)}
    feature_data["target"] = (
        1.5 * feature_data["feature_3"]
        - 0.7 * feature_data["feature_11"]
        + rng.normal(0.0, 0.2, 250)
    )
    result = gradient_boosting_forecast(pd.DataFrame(feature_data))

    assert not result.empty
    assert result["n_estimators"].eq(24).all()
    assert result["max_depth"].eq(2).all()
    assert result["feature"].nunique() <= 24
    assert result["train_size"].max() <= 400


def test_logistic_classifier_handles_multiclass_as_one_vs_rest():
    df = pd.DataFrame(
        {
            "纹饰": ["A", "B", "C", "A", "B", "C", "A", "A"],
            "feature_a": [1.0, 2.1, 3.2, 1.2, 2.0, 3.4, 1.1, 1.3],
            "feature_b": [5.0, 4.0, 3.0, 4.8, 4.1, 3.2, 5.1, 4.9],
        }
    )
    result = logistic_binary_classifier(df)

    assert not result.empty
    assert result["label_column"].iloc[0] == "纹饰"
    assert result["class_count"].iloc[0] == 3
    assert result["binary_target"].iloc[0].endswith("_vs_rest")


def test_logistic_classifier_prefers_semantic_label_over_identifier():
    df = pd.DataFrame(
        {
            "孕妇代码": ["A001", "A001", "A002", "A003", "A004", "A005"],
            "胎儿是否健康": ["是", "否", "是", "是", "否", "是"],
            "年龄": [29, 31, 27, 35, 33, 28],
            "BMI": [21.5, 25.1, 22.3, 24.0, 26.2, 20.9],
        }
    )
    result = logistic_binary_classifier(df)

    assert not result.empty
    assert result["label_column"].iloc[0] == "胎儿是否健康"


def test_correlation_analysis_runs(sample_dataframe):
    result = correlation_analysis(sample_dataframe)
    assert not result.empty


def test_models_do_not_mutate_input(sample_dataframe):
    snapshot = sample_dataframe.copy(deep=True)
    linear_trend_forecast(sample_dataframe)
    entropy_weights(sample_dataframe)
    topsis_rank(sample_dataframe)
    demand_capacity_gap(sample_dataframe)
    pd.testing.assert_frame_equal(sample_dataframe, snapshot)
