"""Generated code contract: canonical types for run_summary.json and execution_manifest.json.

Defines the shape every generated ``baseline_analysis.py`` must produce,
so the writing agent, review agent, and exporters never guess file names.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# ── per-model result ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelRunResult:
    """Outcome of a single model inside a generated script."""

    model_id: str
    status: str  # "success" | "skipped" | "failed" | "not_selected"
    table: str | None = None       # relative or absolute path to output CSV
    elapsed_seconds: float = 0.0
    error: str | None = None       # human-readable error when failed/skipped

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # coerce None → null for JSON, keep everything minimal
        return {k: v for k, v in d.items() if k != "model_id" or v is not None}


# ── top-level run summary ─────────────────────────────────────────────────
@dataclass(frozen=True)
class RunSummary:
    """Contract for ``workspace/logs/run_summary.json``."""

    source: str                     # original data file path
    rows: int
    columns: int
    selected_models: list[str]
    model_runs: list[ModelRunResult] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)         # *.png paths
    describe_table: str | None = None
    column_names: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Preserve only non-empty collections and non-None values
        out: dict[str, Any] = {}
        for f in fields(self):
            value = d.get(f.name)
            if isinstance(value, list):
                # always include selected_models even if empty
                if f.name == "selected_models" or value:
                    out[f.name] = value
            elif value is not None:
                out[f.name] = value
        return out


# ── execution manifest ────────────────────────────────────────────────────
@dataclass(frozen=True)
class FileManifestEntry:
    path: str
    sha256: str | None = None
    bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ExecutionManifest:
    """Contract for ``workspace/logs/execution_manifest.json``."""

    python: str                     # sys.version
    executable: str                 # sys.executable
    random_seed: int
    data_files: list[FileManifestEntry] = field(default_factory=list)
    selected_models: list[str] = field(default_factory=list)
    script_sha256: str | None = None
    workspace: str | None = None
    timeout_seconds: int = 120
    max_attempts: int = 3
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k in ("selected_models",)}


# ── validation helpers ────────────────────────────────────────────────────
def validate_run_summary(data: dict[str, Any]) -> list[str]:
    """Return a list of contract violations (empty = valid)."""
    issues: list[str] = []

    for required in ("source", "rows", "columns", "selected_models"):
        if required not in data:
            issues.append(f"missing required key: {required}")

    model_runs = data.get("model_runs", [])
    if not isinstance(model_runs, list):
        issues.append("model_runs must be a list")
    else:
        for idx, run in enumerate(model_runs):
            if not isinstance(run, dict):
                issues.append(f"model_runs[{idx}] is not a dict")
                continue
            if "model_id" not in run:
                issues.append(f"model_runs[{idx}] missing model_id")
            if run.get("status") not in ("success", "skipped", "failed", "not_selected", None):
                issues.append(
                    f"model_runs[{idx}] invalid status: {run.get('status')!r}"
                )

    return issues


def validate_execution_manifest(data: dict[str, Any]) -> list[str]:
    """Return a list of contract violations (empty = valid)."""
    issues: list[str] = []

    for required in ("python", "random_seed"):
        if required not in data:
            issues.append(f"missing required key: {required}")

    data_files = data.get("data_files", [])
    if not isinstance(data_files, list):
        issues.append("data_files must be a list")
    else:
        for idx, entry in enumerate(data_files):
            if not isinstance(entry, dict):
                issues.append(f"data_files[{idx}] is not a dict")
            elif "path" not in entry:
                issues.append(f"data_files[{idx}] missing path")

    return issues


def run_summary_from_file(path: Path) -> dict[str, Any] | None:
    """Read run_summary.json; return None if missing or unparseable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def execution_manifest_from_file(path: Path) -> dict[str, Any] | None:
    """Read execution_manifest.json; return None if missing or unparseable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
