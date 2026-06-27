from __future__ import annotations

import pandas as pd

from tools.validation_runner import (
    feature_ablation,
    perturbation_robustness,
    rolling_origin_backtest,
)


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": list(range(2015, 2025)),
            "demand": [100, 108, 115, 123, 132, 140, 151, 160, 172, 185],
            "cost": [30, 31, 33, 35, 36, 38, 41, 43, 45, 48],
        }
    )


def test_rolling_origin_backtest_executes_real_forecasts():
    result = rolling_origin_backtest(
        _forecast_frame(),
        ["trend_forecast", "smoothing_forecast", "grey_gm11"],
    )

    assert set(result["model_id"]) == {
        "trend_forecast",
        "smoothing_forecast",
        "grey_gm11",
    }
    assert {"mae", "rmse", "mape", "r2"}.issubset(result.columns)
    assert (result["origins"] > 0).all()


def test_robustness_and_ablation_execute_model_again():
    frame = _forecast_frame()

    robustness = perturbation_robustness(
        frame, ["trend_forecast"], perturbation=0.1
    )
    ablation = feature_ablation(frame, "trend_forecast")

    assert not robustness.empty
    assert set(robustness["perturbation_pct"]) == {-10.0, 10.0}
    assert not ablation.empty
    assert "relative_change_pct" in ablation.columns
