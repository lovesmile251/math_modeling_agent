from __future__ import annotations

import logging
import os

from agents.base import (
    A_CODE,
    K_EXECUTION_ERROR,
    K_LAST_REPAIR_APPLIED,
    K_LAST_REPAIR_NOTE,
    Agent,
    WorkflowState,
)
from tools.file_tool import read_text, write_text

log = logging.getLogger("mma.code_repair_agent")


class CodeRepairAgent(Agent):
    name = "code_repair_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        script_path = state.artifacts.get(A_CODE)
        if script_path is None or not script_path.exists():
            state.notes[K_LAST_REPAIR_APPLIED] = "false"
            state.notes[K_LAST_REPAIR_NOTE] = "No generated code file was available."
            return state

        stderr = state.notes.get(K_EXECUTION_ERROR, "")
        script = read_text(script_path)
        repaired = script
        actions: list[str] = []

        if "UnicodeDecodeError" in stderr or "ParserError" in stderr:
            repaired, applied = self._add_csv_fallback_reader(repaired)
            if applied:
                actions.append("Added CSV encoding and delimiter fallback reader.")

        if "No such file or directory" in stderr or "FileNotFoundError" in stderr:
            existing_files = [str(path.resolve()) for path in state.data_files if path.exists()]
            old_line = next((line for line in repaired.splitlines() if line.startswith("DATA_FILES = ")), "")
            new_line = f"DATA_FILES = {existing_files!r}"
            if old_line and old_line != new_line:
                repaired = repaired.replace(old_line, new_line, 1)
                actions.append("Removed missing data files from generated script.")

        if "ModuleNotFoundError" in stderr:
            actions.append("Dependency error detected. Install requirements before rerunning.")

        if repaired != script:
            write_text(script_path, repaired)
            state.notes[K_LAST_REPAIR_APPLIED] = "true"
            state.notes[K_LAST_REPAIR_NOTE] = "; ".join(actions)
            log.info("Repair applied: %s", "; ".join(actions))
            return state

        # Heuristic repair didn't help — try LLM-based repair as fallback.
        if (
            state.llm
            and state.llm.enabled
            and os.environ.get("MMA_ALLOW_LLM_CODE_REPAIR") == "1"
        ):
            log.info("Heuristic repair failed, attempting LLM-based repair.")
            try:
                llm_repaired, applied = self._llm_repair(state, script, stderr)
                if applied:
                    write_text(script_path, llm_repaired)
                    state.notes[K_LAST_REPAIR_APPLIED] = "true"
                    state.notes[K_LAST_REPAIR_NOTE] = "LLM-based code repair applied."
                    return state
            except Exception as exc:
                log.warning("LLM repair failed: %s", exc)
                state.notes["llm_repair_error"] = str(exc)

        state.notes[K_LAST_REPAIR_APPLIED] = "false"
        state.notes[K_LAST_REPAIR_NOTE] = "; ".join(actions) or "No matching repair heuristic."
        return state

    def _add_csv_fallback_reader(self, script: str) -> tuple[str, bool]:
        if "def read_csv_with_fallback" in script:
            return script, False

        helper = '''
def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, sep=None, engine="python")

'''
        marker = "\ndef read_table(path: Path) -> pd.DataFrame:\n"
        if marker not in script:
            return script, False

        repaired = script.replace(marker, "\n" + helper + "def read_table(path: Path) -> pd.DataFrame:\n", 1)
        repaired = repaired.replace(
            'if path.suffix.lower() == ".csv":\n        return pd.read_csv(path)',
            'if path.suffix.lower() == ".csv":\n        return read_csv_with_fallback(path)',
        )
        repaired = repaired.replace(
            'if path.suffix.lower() == ".tsv":\n        return pd.read_csv(path, sep="\\t")',
            'if path.suffix.lower() == ".tsv":\n        return pd.read_csv(path, sep="\\t", encoding="utf-8-sig")',
        )
        return repaired, repaired != script

    def _llm_repair(self, state: WorkflowState, script: str, stderr: str) -> tuple[str, bool]:
        """Use LLM to repair code based on stderr output.

        Returns (repaired_script, True) on success, (original, False) on failure.
        """
        system_prompt = (
            "You are a Python code repair assistant. "
            "Given a Python script that produced an error, "
            "fix the code and return ONLY the corrected Python script. "
            "Do not include explanations, markdown fences, or any text outside the script. "
            "Keep all imports, functions, and structure intact — only fix the error."
        )
        user_input = (
            "The following Python script produced this error:\n\n"
            f"STDERR:\n{stderr}\n\n"
            f"SCRIPT:\n{script}\n\n"
            "Problem context:\n" + state.problem_text.strip() + "\n\n"
            "Return ONLY the corrected Python script."
        )
        try:
            repaired = state.llm.complete(system_prompt, user_input)
        except Exception:
            return script, False

        repaired = repaired.strip()
        # Strip markdown fences if the LLM wrapped the code.
        if repaired.startswith("```"):
            lines = repaired.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            repaired = "\n".join(lines).strip()

        if repaired and repaired != script.strip():
            return repaired, True
        return script, False
