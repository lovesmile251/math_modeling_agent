from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import WorkspaceConfig


def _make_workspace(root: Path) -> WorkspaceConfig:
    config = WorkspaceConfig(
        root=root,
        input_dir=root / "input",
        code_dir=root / "code",
        data_dir=root / "data",
        figures_dir=root / "figures",
        tables_dir=root / "tables",
        paper_dir=root / "paper",
        logs_dir=root / "logs",
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023],
            "demand": [120, 135, 148, 162, 181],
            "cost": [36, 38, 41, 44, 49],
            "capacity": [150, 160, 170, 185, 200],
        }
    )


@pytest.fixture
def temp_workspace(tmp_path: Path) -> WorkspaceConfig:
    """An isolated workspace that is *not* tied to the project root.

    Suitable for tests that only read/write artifacts directly (no subprocess
    execution of generated code).
    """
    return _make_workspace(tmp_path / "workspace")


@pytest.fixture
def project_rooted_workspace():
    """An isolated workspace placed directly under the project root.

    The generated ``baseline_analysis.py`` resolves the project root as
    ``Path(__file__).resolve().parents[2]`` (i.e. the parent of the workspace
    directory). Placing the workspace one level under the project root keeps the
    ``models`` package importable from the executed subprocess.
    """
    root = PROJECT_ROOT / f"_test_ws_{uuid.uuid4().hex[:8]}"
    config = _make_workspace(root)
    try:
        yield config
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def sample_csv(temp_workspace, sample_dataframe) -> Path:
    path = temp_workspace.data_dir / "sample.csv"
    sample_dataframe.to_csv(path, index=False)
    return path
