from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mma.code_runner")

_SAFE_ENV_KEYS = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
}


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_python_script(script_path: Path, cwd: Path, timeout_seconds: int = 120) -> RunResult:
    try:
        resolved_cwd = cwd.resolve(strict=True)
        resolved_script = script_path.resolve(strict=True)
        if not resolved_script.is_relative_to(resolved_cwd):
            raise ValueError("Generated script must be located inside its workspace.")

        temp_dir = resolved_cwd / ".runtime"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_ENV_KEYS}
        env.update(
            {
                "HOME": str(temp_dir),
                "USERPROFILE": str(temp_dir),
                "TEMP": str(temp_dir),
                "TMP": str(temp_dir),
                "MPLCONFIGDIR": str(temp_dir / "matplotlib"),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "MPLBACKEND": "Agg",
                "PYTHONHASHSEED": "0",
                "MMA_GENERATED_CODE": "1",
            }
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(resolved_script)],
                cwd=str(resolved_cwd),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return RunResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        log.warning("Script %s timed out after %ds", script_path.name, timeout_seconds)
        return RunResult(
            returncode=-1,
            stdout=_coerce_output(exc.stdout),
            stderr=_coerce_output(exc.stderr)
            + f"\n[TimeoutExpired after {timeout_seconds}s]",
            timed_out=True,
        )
    except (OSError, ValueError) as exc:
        log.exception("Could not run script %s: %s", script_path.name, exc)
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"[RunnerError] {exc}",
        )


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
