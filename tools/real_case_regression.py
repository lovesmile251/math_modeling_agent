from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.competition_corpus import load_corpus_index
from tools.file_tool import write_text
from tools.real_case_benchmark import (
    evaluate_real_case,
    load_real_case_gold,
    mark_real_case_gold_hidden,
)


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
    hidden_gold_path: Path | None = None,
    redact_hidden_cases: bool = True,
) -> dict[str, Any]:
    cases = {case.case_id: case for case in load_corpus_index(corpus_index_path)}
    public_gold_cases = [gold for gold in load_real_case_gold(gold_path) if gold.case_id in cases]
    hidden_gold_cases = []
    if hidden_gold_path is not None and hidden_gold_path.exists():
        hidden_gold_cases = [
            gold
            for gold in mark_real_case_gold_hidden(load_real_case_gold(hidden_gold_path))
            if gold.case_id in cases
        ]
    gold_cases = (public_gold_cases + hidden_gold_cases)[:limit]
    scores = []
    skipped = []
    for gold in gold_cases:
        case = cases[gold.case_id]
        statement_path = corpus_root / case.statement_path
        if not statement_path.exists():
            skipped.append(
                {
                    "case_id": _case_label(gold.case_id, gold.hidden, redact_hidden_cases),
                    "hidden": gold.hidden,
                    "reason": "statement file missing",
                }
            )
            continue
        scores.append(evaluate_real_case(gold, case, corpus_root))

    average_score = round(sum(score.total for score in scores) / max(len(scores), 1), 2)
    average_task_f1 = round(sum(score.task_f1 for score in scores) / max(len(scores), 1), 4)
    primary_accuracy = round(sum(score.primary_hit for score in scores) / max(len(scores), 1), 4)
    candidate_coverage = round(sum(score.candidate_hit for score in scores) / max(len(scores), 1), 4)
    failures = [
        _failure_reason(score, redact_hidden_cases=redact_hidden_cases)
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
        "public_gold_cases": len(public_gold_cases),
        "hidden_gold_cases": len(hidden_gold_cases),
        "case_count": len(scores),
        "hidden_case_count": sum(score.hidden for score in scores),
        "skipped_count": len(skipped),
        "average_score": average_score,
        "average_task_f1": average_task_f1,
        "primary_model_accuracy": primary_accuracy,
        "candidate_model_coverage": candidate_coverage,
        "answer_expectation_case_count": sum(score.answer_expectation_count > 0 for score in scores),
        "answer_expectation_count": sum(score.answer_expectation_count for score in scores),
        "min_average_score": min_average_score,
        "min_average_task_f1": min_average_task_f1,
        "min_primary_accuracy": min_primary_accuracy,
        "min_candidate_coverage": min_candidate_coverage,
        "hidden_gold_enabled": bool(hidden_gold_cases),
        "redact_hidden_cases": redact_hidden_cases,
        "passed": passed,
        "failures": failures,
        "scores": [
            _score_payload(score, redact_hidden_cases=redact_hidden_cases)
            for score in scores
        ],
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
        f"- public_gold_cases: {summary['public_gold_cases']}",
        f"- hidden_gold_cases: {summary['hidden_gold_cases']}",
        f"- executed_cases: {summary['case_count']}",
        f"- hidden_executed_cases: {summary['hidden_case_count']}",
        f"- skipped_cases: {summary['skipped_count']}",
        f"- average_score: {summary['average_score']}",
        f"- average_task_f1: {summary['average_task_f1']}",
        f"- primary_model_accuracy: {summary['primary_model_accuracy']}",
        f"- candidate_model_coverage: {summary['candidate_model_coverage']}",
        f"- answer_expectation_cases: {summary['answer_expectation_case_count']}",
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


def _failure_reason(score: Any, *, redact_hidden_cases: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if score.task_f1 < 1.0:
        reasons.append(f"task_f1={score.task_f1}")
    if not score.primary_hit:
        reasons.append("primary model missed")
    if not score.candidate_hit:
        reasons.append("top-5 candidate missed")
    case_id = _case_label(score.case_id, bool(getattr(score, "hidden", False)), redact_hidden_cases)
    return {
        "case_id": case_id,
        "hidden": bool(getattr(score, "hidden", False)),
        "reason": "; ".join(reasons) or f"score={score.total}",
        "selected_models": [] if bool(getattr(score, "hidden", False)) and redact_hidden_cases else list(score.selected_models[:5]),
        "expected_tasks": [] if bool(getattr(score, "hidden", False)) and redact_hidden_cases else list(score.expected_tasks),
    }


def _score_payload(score: Any, *, redact_hidden_cases: bool) -> dict[str, Any]:
    payload = score.to_dict()
    if payload.get("hidden") and redact_hidden_cases:
        payload["case_id"] = _case_label(str(score.case_id), True, True)
        payload["expected_tasks"] = []
        payload["actual_tasks"] = []
        payload["selected_models"] = []
    return payload


def _case_label(case_id: str, hidden: bool, redact_hidden_cases: bool) -> str:
    if not hidden or not redact_hidden_cases:
        return case_id
    import hashlib

    return f"hidden:{hashlib.sha256(case_id.encode('utf-8')).hexdigest()[:12]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-3 real-case regression gate.")
    parser.add_argument("--corpus-index", type=Path, default=Path("benchmarks/real_competition_corpus.json"))
    parser.add_argument("--gold", type=Path, default=Path("benchmarks/real_competition_gold.json"))
    parser.add_argument("--corpus-root", type=Path, default=Path("examples/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--hidden-gold", type=Path, default=None)
    parser.add_argument("--show-hidden-cases", action="store_true")
    args = parser.parse_args()
    summary = run_real_case_regression(
        corpus_index_path=args.corpus_index,
        gold_path=args.gold,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        limit=args.limit,
        hidden_gold_path=args.hidden_gold,
        redact_hidden_cases=not args.show_hidden_cases,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
