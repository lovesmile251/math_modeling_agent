from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.file_tool import write_text
from tools.real_case_benchmark import evaluate_real_case, load_real_case_gold
from tools.competition_corpus import load_corpus_index


def run_real_case_regression(
    corpus_index_path: Path = Path("benchmarks/real_competition_corpus.json"),
    gold_path: Path = Path("benchmarks/real_competition_gold.json"),
    corpus_root: Path = Path("examples/extracted"),
    output_dir: Path = Path("benchmarks/results"),
    *,
    limit: int = 20,
    min_average_score: float = 70.0,
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
    candidate_coverage = round(sum(score.candidate_hit for score in scores) / max(len(scores), 1), 4)
    passed = bool(scores) and average_score >= min_average_score and candidate_coverage >= min_candidate_coverage
    summary = {
        "schema_version": "1.0",
        "requested_cases": len(gold_cases),
        "case_count": len(scores),
        "skipped_count": len(skipped),
        "average_score": average_score,
        "candidate_model_coverage": candidate_coverage,
        "min_average_score": min_average_score,
        "min_candidate_coverage": min_candidate_coverage,
        "passed": passed,
        "scores": [score.to_dict() for score in scores],
        "skipped": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "real_case_regression.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(output_dir / "real_case_regression.md", _format_report(summary))
    return summary


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 第三阶段真实题 Regression",
        "",
        f"- 请求题数：{summary['requested_cases']}",
        f"- 实跑题数：{summary['case_count']}",
        f"- 跳过题数：{summary['skipped_count']}",
        f"- 平均分：{summary['average_score']}",
        f"- 候选覆盖率：{summary['candidate_model_coverage']}",
        f"- 结论：{'通过' if summary['passed'] else '未通过'}",
        "",
        "## 分题",
        "",
        "| case_id | score | candidate_hit | selected_models |",
        "|---|---:|:---:|---|",
    ]
    for item in summary["scores"]:
        lines.append(
            f"| {item['case_id']} | {item['total']:.2f} | "
            f"{'是' if item['candidate_hit'] else '否'} | "
            f"{', '.join(item['selected_models'][:5])} |"
        )
    if summary["skipped"]:
        lines.extend(["", "## 跳过", ""])
        lines.extend(f"- {item['case_id']}: {item['reason']}" for item in summary["skipped"])
    return "\n".join(lines)


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
