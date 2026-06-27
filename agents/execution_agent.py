from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

from agents.base import (
    A_CODE,
    A_EXPERIMENT_REPORT,
    A_EXECUTION_LOG,
    K_EXECUTION_ATTEMPTS,
    K_EXECUTION_ERROR,
    K_EXECUTION_STATUS,
    K_LAST_REPAIR_APPLIED,
    K_LAST_REPAIR_NOTE,
    K_SELECTED_MODEL_IDS,
    Agent,
    WorkflowState,
)
from agents.code_repair_agent import CodeRepairAgent
from tools.code_runner import run_python_script
from tools.file_tool import write_text
from tools.generated_code_quality import check_generated_code
from tools.experiment_runner import build_experiment_report

log = logging.getLogger("mma.execution_agent")


class ExecutionAgent(Agent):
    name = "execution_agent"

    def __init__(self, max_attempts: int = 3, timeout_seconds: int = 120) -> None:
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.repair_agent = CodeRepairAgent()

    def run(self, state: WorkflowState) -> WorkflowState:
        script_path = state.artifacts.get(A_CODE)
        if script_path is None or not script_path.exists():
            log.warning("No code artifact found; cannot execute.")
            state.notes[K_EXECUTION_STATUS] = "failed"
            state.add_error(self.name, "No generated code file to execute.")
            return state

        logs: list[str] = []
        manifest_attempts: list[dict] = []
        final_ok = False

        # ── quality gate: catch syntax/lint errors before execution ──────
        gate = check_generated_code(script_path)
        gate_note = f"quality gate: {gate.summary()}"
        log.info(gate_note)
        logs.append(gate_note)
        if not gate.syntax_ok:
            log.warning("Generated code has syntax errors; sending to repair agent.")
            state.notes[K_EXECUTION_ERROR] = "\n".join(gate.issues)
            state.notes[K_LAST_REPAIR_NOTE] = gate_note
            state = self.repair_agent.run(state)
            if state.notes.get(K_LAST_REPAIR_APPLIED) != "true":
                log.info("Repair agent could not fix syntax errors; aborting.")
                state.notes[K_EXECUTION_STATUS] = "failed"
                state.add_error(self.name, f"Generated code syntax errors: {gate_note}")
                log_path = write_text(state.workspace.logs_dir / "execution.log", "\n".join(logs))
                state.artifacts[A_EXECUTION_LOG] = log_path
                return state
            logs.append(f"repair applied: {state.notes.get(K_LAST_REPAIR_NOTE, '')}")
            gate = check_generated_code(script_path)
            logs.append(f"post-repair quality gate: {gate.summary()}")
            if not gate.passed:
                state.notes[K_EXECUTION_STATUS] = "failed"
                state.notes[K_EXECUTION_ERROR] = "\n".join(gate.issues)
                state.add_error(self.name, "Repaired code failed the quality or security gate.")
                log_path = write_text(
                    state.workspace.logs_dir / "execution.log", "\n".join(logs)
                )
                state.artifacts[A_EXECUTION_LOG] = log_path
                return state
        elif not gate.security_ok:
            log.error("Generated code failed the security gate; refusing execution.")
            state.notes[K_EXECUTION_STATUS] = "failed"
            state.notes[K_EXECUTION_ERROR] = "\n".join(gate.issues)
            state.add_error(self.name, f"Generated code security errors: {gate_note}")
            log_path = write_text(state.workspace.logs_dir / "execution.log", "\n".join(logs))
            state.artifacts[A_EXECUTION_LOG] = log_path
            return state

        for attempt in range(1, self.max_attempts + 1):
            log.info("Execution attempt %d/%d", attempt, self.max_attempts)
            attempt_manifest = {
                "attempt": attempt,
                "script": _file_manifest_entry(script_path),
                "data_files": [_file_manifest_entry(path) for path in state.data_files],
            }
            manifest_attempts.append(attempt_manifest)
            self._write_execution_manifest(state, script_path, manifest_attempts)
            result = run_python_script(
                script_path, cwd=state.workspace.root, timeout_seconds=self.timeout_seconds
            )
            attempt_manifest["returncode"] = result.returncode
            attempt_manifest["timed_out"] = result.timed_out
            self._write_execution_manifest(state, script_path, manifest_attempts)
            attempt_log = [
                f"attempt: {attempt}",
                f"returncode: {result.returncode}",
                f"timed_out: {result.timed_out}",
                "",
                "stdout:",
                result.stdout.strip(),
                "",
                "stderr:",
                result.stderr.strip(),
                "",
            ]
            logs.extend(attempt_log)
            write_text(
                state.workspace.logs_dir / f"execution_attempt_{attempt}.log",
                "\n".join(attempt_log),
            )

            state.notes[K_EXECUTION_ATTEMPTS] = str(attempt)
            state.notes[K_EXECUTION_STATUS] = "success" if result.ok else "failed"
            if result.ok:
                final_ok = True
                state.notes.pop(K_EXECUTION_ERROR, None)
                log.info("Execution succeeded on attempt %d.", attempt)
                report_path, comparison_path = build_experiment_report(
                    state.workspace,
                    state.experiment_plan,
                    state.model_decision,
                )
                state.artifacts[A_EXPERIMENT_REPORT] = report_path
                state.artifacts["experiment_comparison"] = comparison_path
                break

            state.notes[K_EXECUTION_ERROR] = result.stderr
            state.add_error(
                self.name,
                f"Attempt {attempt} failed (rc={result.returncode}, timed_out={result.timed_out})",
            )
            if attempt == self.max_attempts:
                break

            state = self.repair_agent.run(state)
            logs.extend(
                [
                    "repair:",
                    state.notes.get(K_LAST_REPAIR_NOTE, ""),
                    "",
                ]
            )
            if state.notes.get(K_LAST_REPAIR_APPLIED) != "true":
                log.info("Repair agent could not fix the code; stopping retries.")
                break

        log_path = write_text(state.workspace.logs_dir / "execution.log", "\n".join(logs))
        state.artifacts[A_EXECUTION_LOG] = log_path
        state.notes[K_EXECUTION_STATUS] = "success" if final_ok else "failed"
        if not final_ok:
            state.add_error(self.name, f"Execution failed after {attempt} attempt(s).")
        return state

    def _write_execution_manifest(
        self,
        state: WorkflowState,
        script_path: Path,
        attempts: list[dict],
    ) -> None:
        manifest = {
            "python": sys.version,
            "executable": sys.executable,
            "random_seed": 42,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "workspace": str(state.workspace.root),
            "script": _file_manifest_entry(script_path),
            "data_files": [_file_manifest_entry(path) for path in state.data_files],
            "selected_models": state.notes.get(K_SELECTED_MODEL_IDS, "[]"),
            "attempts": attempts,
        }
        write_text(
            state.workspace.logs_dir / "execution_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest_entry(path: Path) -> dict[str, str | int | None]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path) if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else None,
    }
