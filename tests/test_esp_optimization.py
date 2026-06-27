from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models.optimization.esp import esp_operating_optimization, is_esp_operating_frame
from tools.script_builder import build_script


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cement_esp_frame(rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    t = np.arange(rows, dtype=float)
    c_in = 36.0 + 9.0 * np.sin(t / 11.0) + rng.normal(0.0, 1.1, rows)
    q = 462000.0 + 22000.0 * np.sin(t / 17.0 + 0.5) + rng.normal(0.0, 2500.0, rows)
    temp = 126.0 + 6.0 * np.sin(t / 19.0 + 1.1)
    u1 = 58.0 + 11.0 * np.sin(t / 13.0) + rng.normal(0.0, 0.8, rows)
    u2 = 58.5 + 11.0 * np.sin(t / 13.0 + 0.2) + rng.normal(0.0, 0.8, rows)
    u3 = 48.0 + 9.0 * np.sin(t / 15.0 + 0.6) + rng.normal(0.0, 0.6, rows)
    u4 = 48.0 + 9.0 * np.sin(t / 15.0 + 0.9) + rng.normal(0.0, 0.6, rows)
    t1 = 232.0 + 65.0 * np.cos(t / 13.0)
    t2 = 232.0 + 65.0 * np.cos(t / 13.0 + 0.2)
    t3 = 444.0 + 85.0 * np.cos(t / 14.0)
    t4 = 444.0 + 85.0 * np.cos(t / 14.0 + 0.3)
    p = (
        1750.0
        + 18.0 * (u1 - 58.0)
        + 18.0 * (u2 - 58.5)
        + 13.0 * (u3 - 48.0)
        + 13.0 * (u4 - 48.0)
        + 120.0 * np.sqrt(232.0 / t1)
        + 90.0 * np.sqrt(444.0 / t3)
        + rng.normal(0.0, 12.0, rows)
    )
    c_out = np.clip(49.8 - 0.02 * (u1 - 58.0) - 0.02 * (u2 - 58.5) + rng.normal(0.0, 0.08, rows), 48.8, 50.0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-01", periods=rows, freq="min").astype(str),
            "Temp_C": temp,
            "C_in_gNm3": np.clip(c_in, 18.0, None),
            "Q_Nm3h": q,
            "U1_kV": u1,
            "U2_kV": u2,
            "U3_kV": u3,
            "U4_kV": u4,
            "T1_s": t1,
            "T2_s": t2,
            "T3_s": t3,
            "T4_s": t4,
            "C_out_mgNm3": c_out,
            "P_total_kW": p,
        }
    )


def test_esp_operating_optimization_outputs_submission_numbers():
    df = _cement_esp_frame()

    result = esp_operating_optimization(df)

    assert is_esp_operating_frame(df)
    assert not result.empty
    assert {"standard_tightening_summary", "typical_condition_optimum", "differential_strategy"}.issubset(
        set(result["section"])
    )
    optima = result[result["section"] == "typical_condition_optimum"]
    assert {10.0, 5.0}.issubset(set(optima["standard_mgNm3"].astype(float)))
    assert {"U1_kV", "U2_kV", "U3_kV", "U4_kV", "T1_s", "T2_s", "T3_s", "T4_s"}.issubset(result.columns)
    assert optima["predicted_P_total_kW"].notna().all()
    assert (optima["predicted_C_out_mgNm3"].astype(float) <= optima["standard_mgNm3"].astype(float) + 1e-6).all()

    summary = result[result["section"] == "standard_tightening_summary"].iloc[0]
    assert float(summary["energy_increment_pct"]) > 0.0
    assert float(summary["predicted_P_total_kW"]) > float(summary["baseline_P_total_kW"])

    strategies = result[result["section"] == "differential_strategy"]
    assert len(strategies) == 2
    assert strategies["priority_rule"].notna().all()


def test_generated_script_auto_runs_esp_optimization(tmp_path: Path):
    ws_root = tmp_path / "ws"
    for directory in ("figures", "tables", "logs", "data", "code"):
        (ws_root / directory).mkdir(parents=True, exist_ok=True)
    data_path = ws_root / "data" / "Cement_ESP_Data.csv"
    _cement_esp_frame().to_csv(data_path, index=False)

    script = build_script(
        data_files=[str(data_path.resolve())],
        figures_dir=str((ws_root / "figures").resolve()),
        tables_dir=str((ws_root / "tables").resolve()),
        logs_dir=str((ws_root / "logs").resolve()),
        selected_models=["grey_relation"],
        project_root=str(PROJECT_ROOT),
    )
    script_path = ws_root / "code" / "baseline_analysis.py"
    script_path.write_text(script, encoding="utf-8")

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "MPLBACKEND": "Agg",
    }
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ws_root),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    table_path = ws_root / "tables" / "Cement_ESP_Data_esp_optimization.csv"
    assert table_path.exists()
    table = pd.read_csv(table_path)
    assert "standard_tightening_summary" in set(table["section"])
    assert "differential_strategy" in set(table["section"])

    summary = json.loads((ws_root / "logs" / "run_summary.json").read_text(encoding="utf-8"))
    model_runs = summary[0]["model_runs"]
    esp_run = next(run for run in model_runs if run["model_id"] == "esp_optimization")
    assert esp_run["status"] == "success"
    assert Path(esp_run["table"]).name == "Cement_ESP_Data_esp_optimization.csv"
