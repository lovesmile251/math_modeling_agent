from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agents.base import (
    A_CODE,
    A_PAPER,
    A_PAPER_QUALITY,
    K_EXECUTION_ATTEMPTS,
    K_EXECUTION_STATUS,
    K_PAPER_QUALITY_SCORE,
    QUALITY_GATE_NOTE_KEYS,
)
from agents.execution_agent import ExecutionAgent
from app.config import PROJECT_ROOT, WorkspaceConfig
from tools.competition_corpus import CorpusCase, load_corpus_index
from tools.file_tool import (
    MAX_DATA_FILE_BYTES,
    SUPPORTED_DATA_SUFFIXES,
    read_problem_file,
    validate_data_file,
    write_text,
)
from workflows.modeling_workflow import ModelingWorkflow


DIAGNOSTIC_MODEL_IDS = {"error_analysis", "sensitivity_analysis", "model_comparison"}
GRAPH_MODEL_IDS = {
    "graph_shortest_paths",
    "graph_centrality",
    "graph_mst",
    "graph_max_flow",
    "community_detection",
}


@dataclass(frozen=True)
class RealCaseDrillResult:
    case_id: str
    title: str
    year: int
    problem: str
    workspace: str
    elapsed_seconds: float
    execution_status: str
    execution_attempts: int
    selected_models: tuple[str, ...]
    produced_models: tuple[str, ...]
    empty_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    table_count: int
    figure_count: int
    paper_quality_score: int
    error_count: int
    artifacts: dict[str, str]
    score: float
    quality_gates: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_real_case_drill(
    corpus_index_path: Path,
    corpus_root: Path,
    output_dir: Path,
    runs_root: Path,
    case_ids: list[str] | None = None,
    limit: int | None = 3,
    use_llm: bool = False,
    export_formats: list[str] | None = None,
    max_rows_per_file: int | None = 5000,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run real competition cases through the full local workflow.

    This benchmark is intentionally end-to-end: it reads the real statement,
    passes available attachments to the workflow, executes generated code, and
    records whether the run produced result artifacts and a paper draft.
    """
    corpus_root = corpus_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    cases = _select_cases(load_corpus_index(corpus_index_path), case_ids, limit)
    results: list[RealCaseDrillResult] = []
    for case in cases:
        try:
            result = run_single_case_drill(
                case=case,
                corpus_root=corpus_root,
                runs_root=runs_root,
                use_llm=use_llm,
                export_formats=export_formats,
                max_rows_per_file=max_rows_per_file,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            result = _failed_case_result(case, runs_root, exc)
        results.append(result)
        _write_summary(results, output_dir)
    summary = _summarize(results)
    _write_summary(results, output_dir)
    return summary


def run_single_case_drill(
    case: CorpusCase,
    corpus_root: Path,
    runs_root: Path,
    use_llm: bool = False,
    export_formats: list[str] | None = None,
    max_rows_per_file: int | None = 5000,
    timeout_seconds: int = 300,
) -> RealCaseDrillResult:
    statement_path = corpus_root / case.statement_path
    problem_text = read_problem_file(statement_path)
    workspace = WorkspaceConfig.from_root(
        runs_root / _safe_run_name(case.case_id),
        project_root=PROJECT_ROOT,
    )
    workspace.ensure_dirs()
    data_files = _stage_case_data_files(case, corpus_root, workspace.data_dir, max_rows_per_file)

    t0 = time.perf_counter()
    workflow = ModelingWorkflow(
        use_llm=use_llm,
        export_formats=export_formats,
        skip_export=not bool(export_formats),
        workspace=workspace,
    )
    _configure_execution_timeout(workflow, timeout_seconds)
    state = workflow.run(problem_text, data_files=data_files)
    elapsed = round(time.perf_counter() - t0, 2)

    feedback = _load_model_feedback(workspace.logs_dir / "model_execution_feedback.json")
    selected_models = tuple(state.model_decision.selected_model_ids if state.model_decision else [])
    artifacts = {
        name: str(path)
        for name, path in state.artifacts.items()
        if path.exists()
    }
    paper_quality_score = _parse_int(state.notes.get(K_PAPER_QUALITY_SCORE), default=0)
    result = RealCaseDrillResult(
        case_id=case.case_id,
        title=case.title,
        year=case.year,
        problem=case.problem,
        workspace=str(workspace.root),
        elapsed_seconds=elapsed,
        execution_status=state.notes.get(K_EXECUTION_STATUS, "unknown"),
        execution_attempts=_parse_int(state.notes.get(K_EXECUTION_ATTEMPTS), default=0),
        selected_models=selected_models,
        produced_models=tuple(feedback["produced_models"]),
        empty_models=tuple(feedback["empty_models"]),
        missing_models=tuple(feedback["missing_models"]),
        table_count=len(list(workspace.tables_dir.glob("*.csv"))),
        figure_count=len(list(workspace.figures_dir.glob("*.png"))),
        paper_quality_score=paper_quality_score,
        error_count=len(state.errors),
        artifacts=artifacts,
        score=0.0,
        quality_gates=_extract_quality_gates(state.notes),
    )
    return _with_score(result)


def _select_cases(
    cases: list[CorpusCase],
    case_ids: list[str] | None,
    limit: int | None,
) -> list[CorpusCase]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise ValueError("Unknown case id(s): " + ", ".join(missing))
        return selected
    if limit is None:
        return cases
    return cases[: max(limit, 0)]


def _write_summary(results: list[RealCaseDrillResult], output_dir: Path) -> dict[str, Any]:
    summary = _summarize(results)
    write_text(
        output_dir / "real_case_drill.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    write_text(output_dir / "real_case_drill.md", _format_report(summary))
    return summary


def _failed_case_result(case: CorpusCase, runs_root: Path, exc: Exception) -> RealCaseDrillResult:
    workspace = runs_root / _safe_run_name(case.case_id)
    return RealCaseDrillResult(
        case_id=case.case_id,
        title=case.title,
        year=case.year,
        problem=case.problem,
        workspace=str(workspace),
        elapsed_seconds=0.0,
        execution_status="failed",
        execution_attempts=0,
        selected_models=(),
        produced_models=(),
        empty_models=(),
        missing_models=(),
        table_count=0,
        figure_count=0,
        paper_quality_score=0,
        error_count=1,
        artifacts={"error": f"{type(exc).__name__}: {exc}"},
        score=0.0,
        quality_gates={},
    )


def _case_data_files(case: CorpusCase, corpus_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in case.attachment_paths:
        path = corpus_root / relative
        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in SUPPORTED_DATA_SUFFIXES
            and path.stat().st_size <= MAX_DATA_FILE_BYTES
            and _has_reader_for(path)
            and _passes_data_safety_check(path)
        ):
            files.append(path)
    return files


def _stage_case_data_files(
    case: CorpusCase,
    corpus_root: Path,
    staging_dir: Path,
    max_rows_per_file: int | None,
) -> list[Path]:
    source_files = _case_data_files(case, corpus_root)
    if max_rows_per_file is None:
        return source_files
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for index, source in enumerate(source_files, start=1):
        try:
            frame = _read_sample_frame(source, max_rows_per_file)
        except (OSError, UnicodeDecodeError, ValueError, ImportError, pd.errors.ParserError):
            continue
        target = staging_dir / f"{index:02d}_{_safe_run_name(source.stem)}.csv"
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        staged.append(target)
    return staged


def _read_sample_frame(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=encoding, nrows=max_rows)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, nrows=max_rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig", nrows=max_rows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=max_rows)
    raise ValueError(f"Unsupported data file: {path}")


def _configure_execution_timeout(workflow: ModelingWorkflow, timeout_seconds: int) -> None:
    for agent in workflow.agents:
        if isinstance(agent, ExecutionAgent):
            agent.timeout_seconds = timeout_seconds


def _has_reader_for(path: Path) -> bool:
    if path.suffix.lower() == ".xls":
        return importlib.util.find_spec("xlrd") is not None
    return True


def _passes_data_safety_check(path: Path) -> bool:
    try:
        validate_data_file(path)
    except ValueError:
        return False
    return True


def _load_model_feedback(path: Path) -> dict[str, list[str]]:
    empty = {"produced_models": [], "empty_models": [], "missing_models": []}
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    parsed = {
        key: [
            str(item.get("model_id", ""))
            for item in summary.get(key, [])
            if isinstance(item, dict) and item.get("model_id")
        ]
        for key in empty
    }
    produced = set(parsed["produced_models"])
    ignorable_missing = set(DIAGNOSTIC_MODEL_IDS)
    if produced & GRAPH_MODEL_IDS:
        ignorable_missing.add("scheduling_plan")
    parsed["missing_models"] = [
        model_id
        for model_id in parsed["missing_models"]
        if model_id not in produced and model_id not in ignorable_missing
    ]
    parsed["empty_models"] = [
        model_id
        for model_id in parsed["empty_models"]
        if model_id not in produced and model_id not in ignorable_missing
    ]
    return parsed


def _extract_quality_gates(notes: dict[str, str]) -> dict[str, str]:
    return {
        key: str(notes[key])
        for key in QUALITY_GATE_NOTE_KEYS
        if notes.get(key)
    }


def _gate_score(gates: dict[str, str]) -> float:
    if not gates:
        return 0.5
    failed = sum(str(value).lower() in {"failed", "blocked"} for value in gates.values())
    return max(0.0, 1.0 - failed / max(len(gates), 1))


def _with_score(result: RealCaseDrillResult) -> RealCaseDrillResult:
    model_total = (
        len(result.produced_models)
        + len(result.empty_models)
        + len(result.missing_models)
    )
    model_coverage = len(result.produced_models) / model_total if model_total else 0.0
    artifact_score = sum(
        [
            A_CODE in result.artifacts,
            A_PAPER in result.artifacts,
            A_PAPER_QUALITY in result.artifacts,
            result.table_count > 0,
            result.figure_count > 0,
        ]
    ) / 5
    score = (
        20 * (result.execution_status == "success")
        + 15 * artifact_score
        + 20 * model_coverage
        + 25 * min(max(result.paper_quality_score, 0), 100) / 100
        + 10 * (result.error_count == 0)
        + 10 * _gate_score(result.quality_gates)
    )
    return RealCaseDrillResult(
        **{**result.to_dict(), "score": round(score, 2)}
    )


def _summarize(results: list[RealCaseDrillResult]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "average_score": round(
            sum(item.score for item in results) / max(len(results), 1),
            2,
        ),
        "execution_success_rate": round(
            sum(item.execution_status == "success" for item in results) / max(len(results), 1),
            4,
        ),
        "average_paper_quality": round(
            sum(item.paper_quality_score for item in results) / max(len(results), 1),
            2,
        ),
        "average_tables": round(
            sum(item.table_count for item in results) / max(len(results), 1),
            2,
        ),
        "average_figures": round(
            sum(item.figure_count for item in results) / max(len(results), 1),
            2,
        ),
        "results": [item.to_dict() for item in results],
    }


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 历年国赛真题端到端演练报告",
        "",
        f"- 题目数：{summary['case_count']}",
        f"- 端到端均分：{summary['average_score']}",
        f"- 执行成功率：{summary['execution_success_rate']:.1%}",
        f"- 平均论文质量分：{summary['average_paper_quality']}",
        f"- 平均结果表数量：{summary['average_tables']}",
        f"- 平均图片数量：{summary['average_figures']}",
        "",
        "## 分题结果",
        "",
        "| 题目 | 得分 | 执行 | 论文分 | 表 | 图 | 产出模型 | 缺失模型 | 工作区 |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["results"]:
        lines.append(
            f"| {item['case_id']} | {item['score']:.2f} | "
            f"{'是' if item['execution_status'] == 'success' else '否'} | "
            f"{item['paper_quality_score']} | "
            f"{item['table_count']} | "
            f"{item['figure_count']} | "
            f"{len(item['produced_models'])} | "
            f"{len(item['missing_models'])} | "
            f"`{item['workspace']}` |"
        )
    lines.extend(
        [
            "",
            "## 评分口径",
            "",
            "- 执行成功：25 分。",
            "- 代码、论文、质量报告、结果表、图片等关键产物完整性：20 分。",
            "- 已选模型实际产出覆盖率：20 分。",
            "- 论文质量评分折算：25 分。",
            "- 无工作流错误记录：10 分。",
            "",
            "该报告衡量的是本地端到端交付稳定性，不等价于真实竞赛论文人工评奖结果。",
        ]
    )
    return "\n".join(lines)


def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or ""))
    except (TypeError, ValueError):
        return default


def _safe_run_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real CUMCM cases end-to-end.")
    parser.add_argument(
        "--corpus-index",
        type=Path,
        default=Path("benchmarks/real_competition_corpus.json"),
    )
    parser.add_argument("--corpus-root", type=Path, default=Path("examples/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("workspace/runs/real_case_drill"),
    )
    parser.add_argument("--case", dest="case_ids", nargs="*", help="Specific case ids to run.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of cases to run when --case is not provided. Use -1 for all cases.",
    )
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--export", nargs="*", default=None, help="Optional export formats.")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=5000,
        help="Sample at most this many rows from each attachment for fast drills.",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use original full attachments instead of sampled CSV staging.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Execution timeout for generated code.",
    )
    args = parser.parse_args()

    limit = None if args.limit < 0 else args.limit
    max_rows = None if args.full_data else max(args.max_rows_per_file, 1)
    summary = run_real_case_drill(
        corpus_index_path=args.corpus_index,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        runs_root=args.runs_root,
        case_ids=args.case_ids,
        limit=limit,
        use_llm=args.use_llm,
        export_formats=args.export,
        max_rows_per_file=max_rows,
        timeout_seconds=args.timeout_seconds,
    )
    compact = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
