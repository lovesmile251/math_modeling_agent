from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.competition_corpus import load_corpus_index
from tools.file_tool import write_text
from tools.real_case_benchmark import evaluate_real_case, load_real_case_gold


def run_real_case_regression(
    corpus_index_path: Path = Path("benchmarks/real_competition_corpus.json"),
    gold_path: Path = Path("benchmarks/real_competition_gold.json"),
    corpus_root: Path = Path("examples/extracted"),
    output_dir: Path = Path("benchmarks/results"),
    *,
    limit: int = 20,
    min_average_score: float = 70.0,
    min_average_task_f1: float = 0.70,
    min_primary_accuracy: float = 0.25,
    min_candidate_coverage: float = 0.60,
) -> dict[str, Any]:
    cases = {case.case_id: case for case in load_corpus_index(corpus_index_path)}
    gold_cases = [gold for gold in load_real_case_gold(gold_path) if gold.case_id in cases][:limit]
    scores = []
    skipped = []
    for gold in gold_cases:
        case = cases[gold.case_id]
        statement_path = corpus_root / case.statement_path
        if not statement_path.exists():
            skipped.append({"case_id": gold.case_id, "reason": "statement file missing"})
            continue
        scores.append(evaluate_real_case(gold, case, corpus_root))

    average_score = round(sum(score.total for score in scores) / max(len(scores), 1), 2)
    average_task_f1 = round(sum(score.task_f1 for score in scores) / max(len(scores), 1), 4)
    primary_accuracy = round(sum(score.primary_hit for score in scores) / max(len(scores), 1), 4)
    candidate_coverage = round(sum(score.candidate_hit for score in scores) / max(len(scores), 1), 4)
    failures = [
        _failure_reason(score)
        for score in scores
        if score.total < min_average_score or not score.candidate_hit
    ]
    passed = (
        bool(scores)
        and average_score >= min_average_score
        and average_task_f1 >= min_average_task_f1
        and primary_accuracy >= min_primary_accuracy
        and candidate_coverage >= min_candidate_coverage
    )
    summary = {
        "schema_version": "1.1",
        "requested_cases": len(gold_cases),
        "case_count": len(scores),
        "skipped_count": len(skipped),
        "average_score": average_score,
        "average_task_f1": average_task_f1,
        "primary_model_accuracy": primary_accuracy,
        "candidate_model_coverage": candidate_coverage,
        "min_average_score": min_average_score,
        "min_average_task_f1": min_average_task_f1,
        "min_primary_accuracy": min_primary_accuracy,
        "min_candidate_coverage": min_candidate_coverage,
        "passed": passed,
        "failures": failures,
        "scores": [score.to_dict() for score in scores],
        "skipped": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "real_case_regression.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(output_dir / "real_case_regression.md", _format_report(summary))
    return summary


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 Real Case Regression",
        "",
        f"- requested_cases: {summary['requested_cases']}",
        f"- executed_cases: {summary['case_count']}",
        f"- skipped_cases: {summary['skipped_count']}",
        f"- average_score: {summary['average_score']}",
        f"- average_task_f1: {summary['average_task_f1']}",
        f"- primary_model_accuracy: {summary['primary_model_accuracy']}",
        f"- candidate_model_coverage: {summary['candidate_model_coverage']}",
        f"- passed: {summary['passed']}",
        "",
        "## Cases",
        "",
        "| case_id | score | task_f1 | primary_hit | candidate_hit | selected_models |",
        "|---|---:|---:|:---:|:---:|---|",
    ]
    for item in summary["scores"]:
        lines.append(
            f"| {item['case_id']} | {item['total']:.2f} | "
            f"{item['task_f1']:.3f} | "
            f"{'yes' if item['primary_hit'] else 'no'} | "
            f"{'yes' if item['candidate_hit'] else 'no'} | "
            f"{', '.join(item['selected_models'][:5])} |"
        )
    if summary["skipped"]:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {item['case_id']}: {item['reason']}" for item in summary["skipped"])
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item['case_id']}: {item['reason']}" for item in summary["failures"])
    return "\n".join(lines)


def _failure_reason(score: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if score.task_f1 < 1.0:
        reasons.append(f"task_f1={score.task_f1}")
    if not score.primary_hit:
        reasons.append("primary model missed")
    if not score.candidate_hit:
        reasons.append("top-5 candidate missed")
    return {
        "case_id": score.case_id,
        "reason": "; ".join(reasons) or f"score={score.total}",
        "selected_models": list(score.selected_models[:5]),
        "expected_tasks": list(score.expected_tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-3 real-case regression gate.")
    parser.add_argument("--corpus-index", type=Path, default=Path("benchmarks/real_competition_corpus.json"))
    parser.add_argument("--gold", type=Path, default=Path("benchmarks/real_competition_gold.json"))
    parser.add_argument("--corpus-root", type=Path, default=Path("examples/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    summary = run_real_case_regression(
        corpus_index_path=args.corpus_index,
        gold_path=args.gold,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
