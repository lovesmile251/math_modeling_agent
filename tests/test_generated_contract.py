"""Test generated_contract.py — verify dataclass serialization and validation."""

from __future__ import annotations

import json

from tools.generated_contract import (
    ExecutionManifest,
    FileManifestEntry,
    ModelRunResult,
    RunSummary,
    validate_execution_manifest,
    validate_run_summary,
)


# ── ModelRunResult ─────────────────────────────────────────────────────────
def test_model_run_result_success():
    r = ModelRunResult(model_id="trend_forecast", status="success", table="tables/data_trend_forecast.csv", elapsed_seconds=0.12)
    d = r.to_dict()
    assert d["model_id"] == "trend_forecast"
    assert d["status"] == "success"
    assert d["elapsed_seconds"] == 0.12
    assert d["error"] is None


def test_model_run_result_failed():
    r = ModelRunResult(model_id="bogus", status="failed", error="ImportError: no module 'bogus'", elapsed_seconds=0.05)
    d = r.to_dict()
    assert d["status"] == "failed"
    assert "ImportError" in d["error"]


def test_model_run_result_skipped_empty():
    r = ModelRunResult(model_id="kmeans", status="skipped", error="empty result", elapsed_seconds=0.001)
    d = r.to_dict()
    assert d["table"] is None


# ── RunSummary ─────────────────────────────────────────────────────────────
def test_run_summary_minimal():
    s = RunSummary(source="data.csv", rows=100, columns=8, selected_models=["topsis_rank"])
    d = s.to_dict()
    assert d["source"] == "data.csv"
    assert d["rows"] == 100
    assert d["selected_models"] == ["topsis_rank"]


def test_run_summary_with_model_runs():
    runs = [
        ModelRunResult("m1", "success", "t.csv", 0.1),
        ModelRunResult("m2", "failed", None, 0.2, "crash"),
    ]
    s = RunSummary(source="d.csv", rows=10, columns=3, selected_models=["m1", "m2"], model_runs=runs)
    d = s.to_dict()
    assert len(d["model_runs"]) == 2
    assert d["model_runs"][0]["status"] == "success"
    assert d["model_runs"][1]["status"] == "failed"


def test_run_summary_serializable():
    runs = [ModelRunResult("m1", "success", "t.csv", 0.1)]
    s = RunSummary(source="d.csv", rows=10, columns=3, selected_models=["m1"], model_runs=runs)
    raw = json.dumps(s.to_dict(), ensure_ascii=False)
    parsed = json.loads(raw)
    assert parsed["source"] == "d.csv"


# ── FileManifestEntry / ExecutionManifest ──────────────────────────────────
def test_file_manifest_entry():
    e = FileManifestEntry(path="data/data.csv", sha256="abc123", bytes=1024)
    d = e.to_dict()
    assert d["path"] == "data/data.csv"
    assert d["sha256"] == "abc123"
    assert d["bytes"] == 1024


def test_execution_manifest_minimal():
    m = ExecutionManifest(python="3.11", executable="/usr/bin/python", random_seed=42)
    d = m.to_dict()
    assert d["random_seed"] == 42
    assert d["python"] == "3.11"


def test_execution_manifest_with_files():
    m = ExecutionManifest(
        python="3.11",
        executable="/usr/bin/python",
        random_seed=42,
        data_files=[FileManifestEntry("data/d.csv", "sha", 100)],
        selected_models=["m1"],
        script_sha256="def456",
        workspace="/tmp/ws",
    )
    d = m.to_dict()
    assert d["random_seed"] == 42
    assert len(d["data_files"]) == 1
    assert d["selected_models"] == ["m1"]
    assert d["script_sha256"] == "def456"


# ── validation ─────────────────────────────────────────────────────────────
def test_validate_run_summary_valid():
    data = {
        "source": "d.csv",
        "rows": 10,
        "columns": 3,
        "selected_models": ["m1"],
        "model_runs": [{"model_id": "m1", "status": "success", "table": "t.csv", "elapsed_seconds": 0.1, "error": None}],
    }
    assert validate_run_summary(data) == []


def test_validate_run_summary_missing_required():
    assert len(validate_run_summary({"rows": 1, "columns": 2})) > 0


def test_validate_run_summary_bad_status():
    data = {
        "source": "d.csv",
        "rows": 10,
        "columns": 3,
        "selected_models": ["m1"],
        "model_runs": [{"model_id": "m1", "status": "crashed_hard"}],
    }
    issues = validate_run_summary(data)
    assert any("invalid status" in i for i in issues)


def test_validate_execution_manifest_valid():
    data = {
        "python": "3.11",
        "executable": "/usr/bin/python",
        "random_seed": 42,
        "data_files": [{"path": "d.csv"}],
    }
    assert validate_execution_manifest(data) == []


def test_validate_execution_manifest_missing_seed():
    issues = validate_execution_manifest({"python": "3.11"})
    assert any("random_seed" in i for i in issues)
