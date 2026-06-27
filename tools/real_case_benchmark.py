from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.model_selection_crew import ModelSelectionCrew
from tools.competition_corpus import CorpusCase, load_corpus_index
from tools.file_tool import MAX_DATA_FILE_BYTES, read_problem_file, write_text


@dataclass(frozen=True)
class RealCaseGold:
    case_id: str
    expected_task_types: tuple[str, ...]
    acceptable_primary_models: tuple[str, ...]


@dataclass(frozen=True)
class RealCaseScore:
    case_id: str
    total: float
    task_f1: float
    primary_hit: bool
    candidate_hit: bool
    expected_tasks: tuple[str, ...]
    actual_tasks: tuple[str, ...]
    selected_models: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_real_case_gold(path: Path) -> list[RealCaseGold]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RealCaseGold(
            case_id=str(item["case_id"]),
            expected_task_types=tuple(item["expected_task_types"]),
            acceptable_primary_models=tuple(item["acceptable_primary_models"]),
        )
        for item in payload
    ]


def evaluate_real_case(
    gold: RealCaseGold,
    case: CorpusCase,
    corpus_root: Path,
) -> RealCaseScore:
    statement_path = corpus_root / case.statement_path
    problem_text = read_problem_file(statement_path)
    data_files = [
        corpus_root / relative
        for relative in case.attachment_paths
        if (corpus_root / relative).exists()
        and (corpus_root / relative).stat().st_size <= MAX_DATA_FILE_BYTES
    ]
    result = ModelSelectionCrew().run(problem_text, data_files, [])
    actual_tasks = tuple(sorted({task.task_type for task in result.tasks}))
    selected_models = tuple(item.model_id for item in result.selected)

    expected = set(gold.expected_task_types)
    actual = set(actual_tasks)
    precision = len(expected & actual) / max(len(actual), 1)
    recall = len(expected & actual) / max(len(expected), 1)
    task_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    primary_hit = bool(
        selected_models
        and selected_models[0] in set(gold.acceptable_primary_models)
    )
    candidate_hit = bool(
        set(selected_models[:5]) & set(gold.acceptable_primary_models)
    )
    total = round(60 * task_f1 + 25 * primary_hit + 15 * candidate_hit, 2)
    return RealCaseScore(
        case_id=gold.case_id,
        total=total,
        task_f1=round(task_f1, 4),
        primary_hit=primary_hit,
        candidate_hit=candidate_hit,
        expected_tasks=gold.expected_task_types,
        actual_tasks=actual_tasks,
        selected_models=selected_models,
    )


def run_real_case_benchmark(
    corpus_index_path: Path,
    gold_path: Path,
    corpus_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cases = {case.case_id: case for case in load_corpus_index(corpus_index_path)}
    gold_cases = load_real_case_gold(gold_path)
    scores = [
        evaluate_real_case(gold, cases[gold.case_id], corpus_root)
        for gold in gold_cases
        if gold.case_id in cases
    ]
    summary = {
        "case_count": len(scores),
        "average_score": round(
            sum(score.total for score in scores) / max(len(scores), 1),
            2,
        ),
        "average_task_f1": round(
            sum(score.task_f1 for score in scores) / max(len(scores), 1),
            4,
        ),
        "primary_model_accuracy": round(
            sum(score.primary_hit for score in scores) / max(len(scores), 1),
            4,
        ),
        "candidate_model_coverage": round(
            sum(score.candidate_hit for score in scores) / max(len(scores), 1),
            4,
        ),
        "scores": [score.to_dict() for score in scores],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "real_case_benchmark.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    write_text(output_dir / "real_case_benchmark.md", _format_report(summary))
    return summary


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 历年国赛真题盲测报告",
        "",
        f"- 题目数：{summary['case_count']}",
        f"- 综合均分：{summary['average_score']}",
        f"- 任务识别 F1：{summary['average_task_f1']}",
        f"- 首选模型命中率：{summary['primary_model_accuracy']}",
        f"- 前五候选覆盖率：{summary['candidate_model_coverage']}",
        "",
        "## 分题结果",
        "",
        "| 题目 | 得分 | Task F1 | 首选命中 | 候选覆盖 |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for item in summary["scores"]:
        lines.append(
            f"| {item['case_id']} | {item['total']:.2f} | "
            f"{item['task_f1']:.3f} | "
            f"{'是' if item['primary_hit'] else '否'} | "
            f"{'是' if item['candidate_hit'] else '否'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real CUMCM blind benchmark.")
    parser.add_argument(
        "--corpus-index",
        type=Path,
        default=Path("benchmarks/real_competition_corpus.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("benchmarks/real_competition_gold.json"),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("examples/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
    )
    args = parser.parse_args()
    summary = run_real_case_benchmark(
        args.corpus_index,
        args.gold,
        args.corpus_root,
        args.output_dir,
    )
    compact = {key: value for key, value in summary.items() if key != "scores"}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
