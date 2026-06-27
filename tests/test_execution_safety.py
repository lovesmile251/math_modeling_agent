from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from agents.base import A_CODE, K_SELECTED_MODEL_IDS, WorkflowState
from agents.coding_agent import CodingAgent
from agents.execution_agent import ExecutionAgent
from app.config import WorkspaceConfig
from tools.code_runner import run_python_script
from tools.generated_code_quality import check_generated_code
from tests.conftest import PROJECT_ROOT
from tools.code_runner import run_python_script


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_run_python_script_timeout_preserves_text_output(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text(
        "from __future__ import annotations\n"
        "import time\n"
        "print('started', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )

    result = run_python_script(script, cwd=tmp_path, timeout_seconds=1)

    assert result.timed_out is True
    assert result.returncode == -1
    assert "started" in result.stdout
    assert "[TimeoutExpired after 1s]" in result.stderr


def test_execution_manifest_tracks_repaired_script_hash(temp_workspace):
    data_file = temp_workspace.data_dir / "existing.txt"
    data_file.write_text("ok", encoding="utf-8")
    missing_file = temp_workspace.data_dir / "missing.txt"
    script_path = temp_workspace.code_dir / "baseline_analysis.py"
    script_path.write_text(
        "import pandas as pd\n"
        f"DATA_FILES = {[str(missing_file)]!r}\n"
        "pd.read_csv(DATA_FILES[0])\n",
        encoding="utf-8",
    )
    original_hash = _sha256_file(script_path)
    state = WorkflowState(
        problem_text="test",
        data_files=[data_file],
        workspace=temp_workspace,
    )
    state.artifacts[A_CODE] = script_path

    state = ExecutionAgent(max_attempts=2, timeout_seconds=10).run(state)

    final_hash = _sha256_file(script_path)
    manifest = json.loads(
        (temp_workspace.logs_dir / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert state.notes["execution_status"] == "success"
    assert final_hash != original_hash
    assert manifest["script"]["sha256"] == final_hash
    assert [attempt["script"]["sha256"] for attempt in manifest["attempts"]] == [
        original_hash,
        final_hash,
    ]
    assert [attempt["returncode"] for attempt in manifest["attempts"]] == [1, 0]


def test_generated_script_imports_from_isolated_run_workspace(sample_dataframe):
    workspace = WorkspaceConfig.isolated_run(PROJECT_ROOT, "pytest-isolated-import")
    workspace.ensure_dirs()
    try:
        data_file = workspace.data_dir / "sample.csv"
        data_file.write_text(sample_dataframe.to_csv(index=False), encoding="utf-8")
        state = WorkflowState(
            problem_text="forecast future demand",
            data_files=[data_file],
            workspace=workspace,
        )
        state.notes[K_SELECTED_MODEL_IDS] = json.dumps(["trend_forecast"])

        state = CodingAgent().run(state)
        state = ExecutionAgent(max_attempts=1, timeout_seconds=30).run(state)

        manifest = json.loads(
            (workspace.logs_dir / "execution_manifest.json").read_text(encoding="utf-8")
        )
        script_text = state.artifacts[A_CODE].read_text(encoding="utf-8")
        assert state.notes["execution_status"] == "success"
        assert f"PROJECT_ROOT = Path({str(PROJECT_ROOT.resolve())!r})" in script_text
        assert manifest["workspace"] == str(workspace.root)
        assert (workspace.logs_dir / "run_summary.json").exists()
    finally:
        shutil.rmtree(workspace.root, ignore_errors=True)


def test_generated_code_security_gate_blocks_command_execution(tmp_path):
    script = tmp_path / "malicious.py"
    script.write_text("import subprocess\nsubprocess.run(['whoami'])\n", encoding="utf-8")

    result = check_generated_code(script)

    assert result.security_ok is False
    assert any("import 'subprocess' is not allowed" in issue for issue in result.issues)


def test_runner_does_not_expose_api_keys(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    code_dir = workspace / "code"
    code_dir.mkdir(parents=True)
    script = code_dir / "env_check.py"
    script.write_text(
        "import os\nprint(os.environ.get('OPENAI_API_KEY', 'missing'))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    result = run_python_script(script, workspace, timeout_seconds=10)

    assert result.ok
    assert result.stdout.strip() == "missing"
