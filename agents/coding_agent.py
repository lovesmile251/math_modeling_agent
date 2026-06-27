from __future__ import annotations

import json

import pandas as pd

from agents.base import Agent, K_PROBLEM_TYPE, K_SELECTED_MODEL_IDS, WorkflowState
from models.social_network.campus import is_social_network_problem
from tools.file_tool import write_text
from tools.model_registry import registered_model_ids


class CodingAgent(Agent):
    name = "coding_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        if self._is_social_network(state):
            script = self._build_social_network_script(state)
            state.notes[K_PROBLEM_TYPE] = "social_network"
        else:
            script = self._build_script(state)
            state.notes[K_PROBLEM_TYPE] = "general"
        script_path = state.workspace.code_dir / "baseline_analysis.py"
        write_text(script_path, script)
        state.artifacts["code"] = script_path
        return state

    def _is_social_network(self, state: WorkflowState) -> bool:
        columns: list[str] = []
        for path in state.data_files:
            try:
                if path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path, nrows=5)
                elif path.suffix.lower() == ".tsv":
                    frame = pd.read_csv(path, sep="\t", nrows=5)
                elif path.suffix.lower() in {".xlsx", ".xls"}:
                    frame = pd.read_excel(path, nrows=5)
                else:
                    continue
                columns.extend(str(column) for column in frame.columns)
            except (
                ImportError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                ValueError,
                OSError,
            ):
                continue
        if not columns:
            return False
        return is_social_network_problem(state.problem_text, columns)

    def _build_social_network_script(self, state: WorkflowState) -> str:
        data_files = [str(path.resolve()) for path in state.data_files]
        figures_dir = str(state.workspace.figures_dir.resolve())
        tables_dir = str(state.workspace.tables_dir.resolve())
        logs_dir = str(state.workspace.logs_dir.resolve())
        project_root = str(state.workspace.effective_project_root.resolve())
        problem_text = state.problem_text
        return f'''from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path({project_root!r})
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.social_network.campus import run_campus_social_analysis


DATA_FILES = {data_files!r}
FIGURES_DIR = Path({figures_dir!r})
TABLES_DIR = Path({tables_dir!r})
LOGS_DIR = Path({logs_dir!r})
PROBLEM_TEXT = {problem_text!r}
SELECTED_MODELS = [
    "community_detection",
    "friend_recommendation",
    "information_propagation",
    "influence_maximization",
]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    return pd.read_csv(path, sep=None, engine="python")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_with_fallback(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\\t", encoding="utf-8-sig")
    if suffix in {{".xlsx", ".xls"}}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file: {{path}}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for raw_path in DATA_FILES:
        path = Path(raw_path)
        df = read_table(path)
        stem = path.stem

        for stale in list(TABLES_DIR.glob(f"{{stem}}_*.csv")) + list(FIGURES_DIR.glob(f"{{stem}}_*.png")):
            stale.unlink()

        describe_path = TABLES_DIR / f"{{stem}}_describe.csv"
        df.describe(include="all").transpose().to_csv(describe_path, encoding="utf-8-sig")

        outputs = run_campus_social_analysis(df, FIGURES_DIR, TABLES_DIR, stem, PROBLEM_TEXT)
        model_outputs = {{name: path_ for name, path_ in outputs.items() if str(path_).endswith(".csv")}}
        model_runs = []
        for model_id in SELECTED_MODELS:
            table_path = model_outputs.get(model_id)
            model_runs.append(
                {{
                    "model_id": model_id,
                    "status": "success" if table_path else "skipped",
                    "table": table_path,
                    "elapsed_seconds": 0.0,
                    "error": None if table_path else "no output table produced",
                }}
            )
        charts = [path_ for name, path_ in outputs.items() if str(path_).endswith(".png")]

        summaries.append(
            {{
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
                "missing_values": {{col: int(df[col].isna().sum()) for col in df.columns}},
                "numeric_columns": list(df.select_dtypes(include="number").columns),
                "selected_models": SELECTED_MODELS,
                "model_outputs": model_outputs,
                "model_runs": model_runs,
                "source": str(path),
                "charts": charts,
                "describe_table": str(describe_path),
            }}
        )

    summary_path = LOGS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {{summary_path}}")


if __name__ == "__main__":
    main()
'''

    def _build_script(self, state: WorkflowState) -> str:
        from tools.script_builder import build_script

        return build_script(
            data_files=[str(path.resolve()) for path in state.data_files],
            figures_dir=str(state.workspace.figures_dir.resolve()),
            tables_dir=str(state.workspace.tables_dir.resolve()),
            logs_dir=str(state.workspace.logs_dir.resolve()),
            selected_models=self._selected_models(state),
            project_root=str(state.workspace.effective_project_root.resolve()),
        )

    ALWAYS_ON_MODELS = ["error_analysis", "sensitivity_analysis", "model_comparison"]

    def _selected_models(self, state: WorkflowState) -> list[str]:
        fallback = ["trend_forecast", "entropy_weights", "topsis_rank", "capacity_gap"]
        # Dynamically resolve valid model IDs from the registry — no hardcoded list.
        valid_models = registered_model_ids()
        raw = state.notes.get(K_SELECTED_MODEL_IDS)
        if not raw:
            return self._with_always_on(fallback)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._with_always_on(fallback)
        if not isinstance(parsed, list):
            return self._with_always_on(fallback)
        selected = [str(item) for item in parsed if str(item) in valid_models]
        return self._with_always_on(selected)

    def _with_always_on(self, models: list[str]) -> list[str]:
        merged = list(models)
        for model_id in self.ALWAYS_ON_MODELS:
            if model_id not in merged:
                merged.append(model_id)
        return merged
