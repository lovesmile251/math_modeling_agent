from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.base import (
    K_EXPORT_QUALITY_GATE,
    K_INNOVATION_EVIDENCE_GATE,
    K_STRONG_BASELINE_GATE,
    K_TASK_TRACEABILITY_GATE,
)
from tools.answer_correctness import audit_workspace_correctness, load_gold_expectations
from tools.file_tool import write_text
from tools.real_case_drill import run_real_case_drill


AWARD_TABLE_TARGET = 12
AWARD_FIGURE_TARGET = 8
AWARD_PAPER_TARGET = 90


def run_contest_simulation(
    *,
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
    time_budget_hours: float = 6.0,
    gold_expectations_path: Path | None = None,
) -> dict[str, Any]:
    """Run real cases and score them with a contest-style blind review rubric."""
    drill_summary = run_real_case_drill(
        corpus_index_path=corpus_index_path,
        corpus_root=corpus_root,
        output_dir=output_dir,
        runs_root=runs_root,
        case_ids=case_ids,
        limit=limit,
        use_llm=use_llm,
        export_formats=export_formats,
        max_rows_per_file=max_rows_per_file,
        timeout_seconds=timeout_seconds,
    )
    contest_summary = evaluate_contest_readiness(
        drill_summary,
        time_budget_seconds=max(int(time_budget_hours * 3600), 1),
        gold_expectations=load_gold_expectations(gold_expectations_path),
    )
    write_contest_summary(contest_summary, output_dir)
    return contest_summary


def evaluate_contest_readiness(
    drill_summary: dict[str, Any],
    *,
    time_budget_seconds: int,
    gold_expectations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    results = [
        _score_case(
            item,
            time_budget_seconds=time_budget_seconds,
            gold_expectation=(gold_expectations or {}).get(str(item.get("case_id", ""))),
        )
        for item in drill_summary.get("results", [])
        if isinstance(item, dict)
    ]
    average_contest_score = _average([item["contest_score"] for item in results])
    average_blind_review_score = _average([item["blind_review_score"] for item in results])
    first_prize_ready_rate = _average([item["readiness_band"] == "first_prize_ready" for item in results])
    high_risk_case_count = sum(item["readiness_band"] in {"risky", "not_competitive"} for item in results)
    summary = {
        "case_count": len(results),
        "time_budget_seconds": time_budget_seconds,
        "average_contest_score": average_contest_score,
        "average_blind_review_score": average_blind_review_score,
        "first_prize_ready_rate": round(first_prize_ready_rate, 4),
        "high_risk_case_count": high_risk_case_count,
        "risky_case_count": high_risk_case_count,
        "overall_readiness": _overall_readiness(average_contest_score, first_prize_ready_rate, high_risk_case_count),
        "rubric": {
            "time_management": 7,
            "delivery_completeness": 10,
            "modeling_depth": 16,
            "result_validation": 17,
            "answer_correctness": 15,
            "paper_competitiveness": 18,
            "reproducibility": 10,
            "gate_integrity": 7,
        },
        "results": results,
    }
    return summary


def write_contest_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "contest_simulation.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    write_text(output_dir / "contest_simulation.md", format_contest_report(summary))


def format_contest_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 限时赛制盲审模拟报告",
        "",
        f"- 题目数：{summary['case_count']}",
        f"- 单题时间预算：{summary['time_budget_seconds'] / 3600:.2f} 小时",
        f"- 平均赛制得分：{summary['average_contest_score']}",
        f"- 平均盲审分：{summary['average_blind_review_score']}",
        f"- 一等奖就绪率：{summary['first_prize_ready_rate']:.1%}",
        f"- 高风险题数量：{summary['high_risk_case_count']}",
        f"- 总体判断：{summary['overall_readiness']}",
        "",
        "## 评分维度",
        "",
        "| 维度 | 分值 | 含义 |",
        "|---|---:|---|",
        "| 时间管理 | 7 | 是否能在单题预算内稳定完成，并留出人工检查余量 |",
        "| 交付完整性 | 10 | 代码、论文、质量报告、结果表、图形是否完整 |",
        "| 建模深度 | 16 | 已选核心模型是否实际产出，是否具备多模型支撑 |",
        "| 结果验证 | 17 | 是否有误差、敏感性、模型对比或证据追溯产物 |",
        "| 答案正确性 | 15 | 有金标时，关键数值区间、排序、分类或决策是否符合期望 |",
        "| 论文竞争力 | 18 | 论文质量分、表格、图形是否接近优秀论文结构密度 |",
        "| 可复现性 | 10 | 执行是否成功、是否零错误、是否无需多轮修复 |",
        "| 门禁完整性 | 7 | 导出、任务追踪、强基线、创新证据等硬门禁是否通过或被明确评估 |",
        "",
        "## 分题结果",
        "",
        "| 题目 | 赛制分 | 盲审分 | 分层 | 主要短板 |",
        "|---|---:|---:|---|---|",
    ]
    for item in summary["results"]:
        risks = "；".join(item["risks"][:3]) if item["risks"] else "无"
        lines.append(
            f"| {item['case_id']} | {item['contest_score']:.2f} | "
            f"{item['blind_review_score']:.2f} | {item['readiness_band']} | {risks} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "该报告用于衡量工程化限时交付和论文结构竞争力，不能替代真实国赛评委对创新性、问题理解深度和答案正确性的人工判断。",
        ]
    )
    return "\n".join(lines)


def _score_case(
    item: dict[str, Any],
    *,
    time_budget_seconds: int,
    gold_expectation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = float(item.get("elapsed_seconds") or 0.0)
    execution_success = item.get("execution_status") == "success"
    paper_quality = _clamp(float(item.get("paper_quality_score") or 0.0), 0.0, 100.0)
    table_count = int(item.get("table_count") or 0)
    figure_count = int(item.get("figure_count") or 0)
    produced_models = list(item.get("produced_models") or [])
    missing_models = list(item.get("missing_models") or [])
    empty_models = list(item.get("empty_models") or [])
    artifacts = item.get("artifacts") if isinstance(item.get("artifacts"), dict) else {}
    gates = item.get("quality_gates") if isinstance(item.get("quality_gates"), dict) else {}
    table_names = _collect_table_names(item)
    error_count = int(item.get("error_count") or 0)
    attempts = int(item.get("execution_attempts") or 0)
    gate_assessment = _gate_integrity(gates)
    correctness_audit = audit_workspace_correctness(
        Path(str(item.get("workspace", ""))),
        case_id=str(item.get("case_id", "")),
        expectation=gold_expectation,
    )

    time_score = 7 * _time_margin_score(elapsed, time_budget_seconds)
    delivery_score = 10 * _delivery_score(artifacts, table_count, figure_count)
    modeling_score = 16 * _modeling_score(produced_models, missing_models, empty_models)
    validation_score = 17 * _validation_score(produced_models, artifacts, table_names)
    correctness_score = 15 * _answer_correctness_score(correctness_audit)
    paper_score = 18 * _paper_competitiveness_score(paper_quality, table_count, figure_count)
    reproducibility_score = 10 * _reproducibility_score(execution_success, error_count, attempts)
    gate_score = 7 * gate_assessment["score"]
    contest_score = round(
        time_score
        + delivery_score
        + modeling_score
        + validation_score
        + correctness_score
        + paper_score
        + reproducibility_score
        + gate_score,
        2,
    )
    blind_review_score = round(
        0.25 * paper_quality
        + 0.18 * min(100.0, 100.0 * _modeling_score(produced_models, missing_models, empty_models))
        + 0.16 * min(100.0, 100.0 * _validation_score(produced_models, artifacts, table_names))
        + 0.20 * min(100.0, 100.0 * _answer_correctness_score(correctness_audit))
        + 0.11 * min(100.0, 100.0 * _paper_artifact_density(table_count, figure_count))
        + 0.10 * min(100.0, 100.0 * gate_assessment["score"]),
        2,
    )
    risks = _case_risks(
        elapsed=elapsed,
        time_budget_seconds=time_budget_seconds,
        paper_quality=paper_quality,
        table_count=table_count,
        figure_count=figure_count,
        missing_models=missing_models,
        error_count=error_count,
        blind_review_score=blind_review_score,
        gate_failures=gate_assessment["failures"],
        unknown_gates=gate_assessment["unknown"],
        correctness_audit=correctness_audit,
    )
    return {
        "case_id": str(item.get("case_id", "")),
        "title": str(item.get("title", "")),
        "elapsed_seconds": elapsed,
        "contest_score": contest_score,
        "blind_review_score": blind_review_score,
        "readiness_band": _readiness_band(contest_score, blind_review_score, risks),
        "dimension_scores": {
            "time_management": round(time_score, 2),
            "delivery_completeness": round(delivery_score, 2),
            "modeling_depth": round(modeling_score, 2),
            "result_validation": round(validation_score, 2),
            "answer_correctness": round(correctness_score, 2),
            "paper_competitiveness": round(paper_score, 2),
            "reproducibility": round(reproducibility_score, 2),
            "gate_integrity": round(gate_score, 2),
        },
        "answer_correctness_audit": correctness_audit,
        "quality_gates": gates,
        "risks": risks,
        "workspace": str(item.get("workspace", "")),
    }


def _time_margin_score(elapsed_seconds: float, budget_seconds: int) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    ratio = elapsed_seconds / budget_seconds
    if ratio <= 0.70:
        return 1.0
    if ratio <= 1.0:
        return max(0.6, (1.0 - ratio) / 0.30)
    return max(0.0, 0.6 - min((ratio - 1.0) / 0.50, 0.6))


def _delivery_score(artifacts: dict[str, Any], table_count: int, figure_count: int) -> float:
    keys = {
        "code": "code" in artifacts,
        "paper": "paper" in artifacts,
        "paper_quality": "paper_quality" in artifacts,
        "tables": table_count > 0,
        "figures": figure_count > 0,
    }
    return sum(keys.values()) / len(keys)


def _modeling_score(produced_models: list[str], missing_models: list[str], empty_models: list[str]) -> float:
    total = len(produced_models) + len(missing_models) + len(empty_models)
    if total == 0:
        return 0.0
    coverage = len(produced_models) / total
    breadth_bonus = min(len(set(produced_models)) / 3, 1.0)
    return _clamp(0.75 * coverage + 0.25 * breadth_bonus, 0.0, 1.0)


def _collect_table_names(item: dict[str, Any]) -> list[str]:
    workspace = Path(str(item.get("workspace") or ""))
    tables_dir = workspace / "tables"
    if not tables_dir.exists():
        return []
    return [path.name.lower() for path in tables_dir.glob("*.csv")]


def _validation_score(
    produced_models: list[str],
    artifacts: dict[str, Any],
    table_names: list[str] | None = None,
) -> float:
    produced = set(produced_models)
    table_names = table_names or []

    def has_table(fragment: str) -> bool:
        return any(fragment in name for name in table_names)

    checks = [
        "error_analysis" in produced or has_table("data_quality_scorecard"),
        "sensitivity_analysis" in produced or has_table("feature_summary"),
        "model_comparison" in produced or has_table("correlation_pairs") or has_table("pair_frequency"),
        "claim_evidence_map" in artifacts or "result_registry" in artifacts,
        has_table("missingness_summary") or has_table("categorical_frequency") or has_table("column_type_summary"),
    ]
    return sum(checks) / len(checks)


def _paper_competitiveness_score(paper_quality: float, table_count: int, figure_count: int) -> float:
    quality = paper_quality / 100
    density = _paper_artifact_density(table_count, figure_count)
    award_quality = min(paper_quality / AWARD_PAPER_TARGET, 1.0)
    return _clamp(0.55 * quality + 0.25 * density + 0.20 * award_quality, 0.0, 1.0)


def _paper_artifact_density(table_count: int, figure_count: int) -> float:
    table_score = min(table_count / AWARD_TABLE_TARGET, 1.0)
    figure_score = min(figure_count / AWARD_FIGURE_TARGET, 1.0)
    return 0.5 * table_score + 0.5 * figure_score


def _answer_correctness_score(audit: dict[str, Any]) -> float:
    if not audit.get("applicable"):
        return 1.0
    return _clamp(float(audit.get("pass_rate") or 0.0), 0.0, 1.0)


def _reproducibility_score(execution_success: bool, error_count: int, attempts: int) -> float:
    if not execution_success:
        return 0.0
    score = 1.0
    if error_count:
        score -= 0.45
    if attempts > 1:
        score -= min(0.25, 0.08 * (attempts - 1))
    return _clamp(score, 0.0, 1.0)


def _gate_integrity(gates: dict[str, Any]) -> dict[str, Any]:
    expected = (
        K_EXPORT_QUALITY_GATE,
        K_TASK_TRACEABILITY_GATE,
        K_STRONG_BASELINE_GATE,
        K_INNOVATION_EVIDENCE_GATE,
    )
    if not gates:
        return {"score": 0.5, "failures": [], "unknown": list(expected)}

    failures = [
        key
        for key, value in gates.items()
        if str(value).lower() in {"failed", "blocked", "fail"}
    ]
    unknown = [key for key in expected if key not in gates]
    passed = [
        key
        for key, value in gates.items()
        if str(value).lower() in {"passed", "ok", "success", "completed"}
    ]
    denominator = max(len(set(expected) | set(gates)), 1)
    score = (len(passed) + 0.5 * len(unknown)) / denominator
    score -= len(failures) / denominator
    return {
        "score": _clamp(score, 0.0, 1.0),
        "failures": failures,
        "unknown": unknown,
    }


def _case_risks(
    *,
    elapsed: float,
    time_budget_seconds: int,
    paper_quality: float,
    table_count: int,
    figure_count: int,
    missing_models: list[str],
    error_count: int,
    blind_review_score: float,
    gate_failures: list[str],
    unknown_gates: list[str],
    correctness_audit: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    if elapsed > 0.85 * time_budget_seconds:
        risks.append("时间余量不足")
    if missing_models:
        risks.append("存在核心模型未产出：" + ", ".join(missing_models[:3]))
    if error_count:
        risks.append("工作流存在错误记录")
    if gate_failures:
        risks.append("Gate failed: " + ", ".join(gate_failures[:3]))
    if unknown_gates:
        risks.append("Gate not evaluated: " + ", ".join(unknown_gates[:3]))
    if correctness_audit.get("applicable") and not correctness_audit.get("passed"):
        risks.append("answer correctness expectations failed")
    if paper_quality < AWARD_PAPER_TARGET:
        risks.append(f"论文质量分低于一等奖冲刺线 {AWARD_PAPER_TARGET}")
    if table_count < AWARD_TABLE_TARGET:
        risks.append(f"结果表数量低于优秀论文中位参考 {AWARD_TABLE_TARGET}")
    if figure_count < AWARD_FIGURE_TARGET:
        risks.append(f"图片数量低于优秀论文中位参考 {AWARD_FIGURE_TARGET}")
    if blind_review_score < 85:
        risks.append("盲审结构分未达到稳健冲奖线")
    return risks


def _readiness_band(contest_score: float, blind_review_score: float, risks: list[str]) -> str:
    if contest_score >= 92 and blind_review_score >= 88 and not risks:
        return "first_prize_ready"
    if contest_score >= 85 and blind_review_score >= 80:
        return "competitive"
    if contest_score >= 75 and blind_review_score >= 70:
        return "risky"
    return "not_competitive"


def _overall_readiness(average_score: float, first_prize_ready_rate: float, risky_case_count: int) -> str:
    if average_score >= 92 and first_prize_ready_rate >= 0.6 and risky_case_count == 0:
        return "具备一等奖冲刺工程基础"
    if average_score >= 85 and risky_case_count <= 1:
        return "具备稳定参赛交付基础，但仍需人工强化创新与论文表达"
    if average_score >= 75:
        return "可完成参赛闭环，但存在明显冲奖风险"
    return "尚不适合作为限时竞赛主力方案"


def _average(values: list[float | bool]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 2)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def format_contest_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 限时赛制盲审模拟报告",
        "",
        f"- 题目数：{summary['case_count']}",
        f"- 单题时间预算：{summary['time_budget_seconds'] / 3600:.2f} 小时",
        f"- 平均赛制得分：{summary['average_contest_score']}",
        f"- 平均盲审分：{summary['average_blind_review_score']}",
        f"- 一等奖就绪率：{summary['first_prize_ready_rate']:.1%}",
        f"- 高风险题数量：{summary['high_risk_case_count']}",
        f"- 总体判断：{summary['overall_readiness']}",
        "",
        "## 评分维度",
        "",
        "| 维度 | 分值 | 含义 |",
        "|---|---:|---|",
        "| 时间管理 | 7 | 是否能在单题预算内稳定完成，并留出人工检查余量 |",
        "| 交付完整性 | 10 | 代码、论文、质量报告、结果表、图形是否完整 |",
        "| 建模深度 | 16 | 核心模型是否实际产出，是否具备多模型支撑 |",
        "| 结果验证 | 17 | 是否有误差、敏感性、模型对比或证据追溯产物 |",
        "| 答案正确性 | 15 | 有金标时，关键数值区间、排序、分类或决策是否符合期望 |",
        "| 论文竞争力 | 18 | 论文质量分、表格、图形是否接近优秀论文结构密度 |",
        "| 可复现性 | 10 | 执行是否成功、是否零错误、是否无需多轮修复 |",
        "| 门禁完整性 | 7 | 导出、任务追踪、强基线、创新证据等硬门禁是否通过或被明确评估 |",
        "",
        "## 分题结果",
        "",
        "| 题目 | 赛制分 | 盲审分 | 分层 | 主要短板 |",
        "|---|---:|---:|---|---|",
    ]
    for item in summary["results"]:
        risks = "；".join(item["risks"][:3]) if item["risks"] else "无"
        lines.append(
            f"| {item['case_id']} | {item['contest_score']:.2f} | "
            f"{item['blind_review_score']:.2f} | {item['readiness_band']} | {risks} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "该报告用于衡量工程化限时交付和论文结构竞争力，不能替代真实国赛评委对创新性、问题理解深度和答案正确性的人工判断。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real cases with a contest-style blind review rubric.")
    parser.add_argument("--corpus-index", type=Path, default=Path("benchmarks/real_competition_corpus.json"))
    parser.add_argument("--corpus-root", type=Path, default=Path("examples/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--runs-root", type=Path, default=Path("workspace/runs/contest_simulation"))
    parser.add_argument("--case", dest="case_ids", nargs="*", help="Specific case ids to run.")
    parser.add_argument("--limit", type=int, default=3, help="Number of cases to run when --case is not provided. Use -1 for all cases.")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--export", nargs="*", default=None, help="Optional export formats.")
    parser.add_argument("--max-rows-per-file", type=int, default=5000)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--time-budget-hours", type=float, default=6.0)
    parser.add_argument("--gold-expectations", type=Path, default=None)
    args = parser.parse_args()

    max_rows = None if args.full_data else max(args.max_rows_per_file, 1)
    limit = None if args.limit < 0 else args.limit
    summary = run_contest_simulation(
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
        time_budget_hours=args.time_budget_hours,
        gold_expectations_path=args.gold_expectations,
    )
    compact = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
