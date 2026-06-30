from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.file_tool import write_text


EXPECTED_GATES = (
    "export_quality_gate",
    "task_traceability_gate",
    "strong_baseline_gate",
    "innovation_evidence_gate",
    "paper_evidence_gate",
)
REQUIRED_ARTIFACTS = (
    "code",
    "paper",
    "paper_quality",
    "claim_evidence_map",
    "paper_evidence_audit",
)
AWARD_TABLE_TARGET = 12
AWARD_FIGURE_TARGET = 8
AWARD_PAPER_TARGET = 90

ROOT_CAUSE_RULES: tuple[dict[str, Any], ...] = (
    {
        "cluster": "execution_and_model_outputs",
        "categories": {"execution_failure", "workflow_errors", "model_output_gap"},
        "gates": set(),
        "recommended_phase": "code_plan",
        "action": "Regenerate the code plan and executable model outputs before improving the paper.",
        "severity": 5,
    },
    {
        "cluster": "experiment_and_innovation_evidence",
        "categories": set(),
        "gates": {"strong_baseline_gate", "innovation_evidence_gate"},
        "recommended_phase": "experiment_plan",
        "action": "Add executed baselines, ablations, validation, and innovation evidence.",
        "severity": 4,
    },
    {
        "cluster": "answer_and_evidence_binding",
        "categories": {"answer_correctness_failed"},
        "gates": {"task_traceability_gate"},
        "recommended_phase": "evidence_mapping",
        "action": "Rebuild result analysis, claim evidence mapping, and task-to-paper bindings.",
        "severity": 4,
    },
    {
        "cluster": "paper_evidence_and_award_density",
        "categories": {"weak_paper_quality", "sparse_tables", "sparse_figures", "low_contest_score"},
        "gates": {"paper_evidence_gate", "prewriting_gate_status"},
        "recommended_phase": "section_writing",
        "action": "Rewrite paper sections around table-backed metrics, figures, and contest answers.",
        "severity": 3,
    },
    {
        "cluster": "export_and_gate_coverage",
        "categories": {"artifact_gap", "gate_missing"},
        "gates": {"export_quality_gate"},
        "recommended_phase": "export",
        "action": "Run the export stage and ensure every hard gate emits an artifact and status.",
        "severity": 2,
    },
)


def build_pressure_audit(
    drill_summary: dict[str, Any],
    contest_summary: dict[str, Any] | None = None,
    *,
    time_budget_seconds: int | None = None,
    contest_score_floor: float = 85.0,
) -> dict[str, Any]:
    drill_results = [
        item for item in drill_summary.get("results", []) if isinstance(item, dict)
    ]
    contest_by_case = _contest_results_by_case(contest_summary)
    case_findings = [
        _case_finding(
            item,
            contest_by_case.get(str(item.get("case_id", ""))),
            time_budget_seconds=time_budget_seconds,
            contest_score_floor=contest_score_floor,
        )
        for item in drill_results
    ]
    category_counts = Counter(
        category
        for finding in case_findings
        for category in finding["failure_categories"]
    )
    gate_counts = Counter(
        gate
        for finding in case_findings
        for gate in finding["failed_gates"]
    )
    bottlenecks = _rank_bottlenecks(category_counts, gate_counts)
    root_cause_clusters = _cluster_root_causes(case_findings)
    case_count = len(case_findings)
    return {
        "schema_version": "1.0",
        "case_count": case_count,
        "aggregate": {
            "average_drill_score": _average(item.get("score") for item in drill_results),
            "average_contest_score": _average(
                item.get("contest_score") for item in contest_by_case.values()
            ),
            "execution_success_rate": _rate(
                item.get("execution_status") == "success" for item in drill_results
            ),
            "export_ready_rate": _rate(
                _gate_passed(item.get("quality_gates", {}), "export_quality_gate")
                for item in drill_results
            ),
            "paper_evidence_pass_rate": _rate(
                _gate_passed(item.get("quality_gates", {}), "paper_evidence_gate")
                for item in drill_results
            ),
            "average_paper_quality": _average(
                item.get("paper_quality_score") for item in drill_results
            ),
            "average_tables": _average(item.get("table_count") for item in drill_results),
            "average_figures": _average(item.get("figure_count") for item in drill_results),
        },
        "failure_taxonomy": dict(sorted(category_counts.items())),
        "failed_gate_counts": dict(sorted(gate_counts.items())),
        "bottlenecks": bottlenecks,
        "root_cause_clusters": root_cause_clusters,
        "case_findings": case_findings,
    }


def write_pressure_audit(summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_text(
        output_dir / "pressure_audit.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    md_path = write_text(output_dir / "pressure_audit.md", format_pressure_audit(summary))
    return {"json": json_path, "markdown": md_path}


def format_pressure_audit(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# P3 End-to-End Pressure Audit",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Average drill score: {aggregate.get('average_drill_score', 0)}",
        f"- Average contest score: {aggregate.get('average_contest_score', 0)}",
        f"- Execution success rate: {float(aggregate.get('execution_success_rate', 0)):.1%}",
        f"- Export ready rate: {float(aggregate.get('export_ready_rate', 0)):.1%}",
        f"- Paper evidence pass rate: {float(aggregate.get('paper_evidence_pass_rate', 0)):.1%}",
        f"- Average paper quality: {aggregate.get('average_paper_quality', 0)}",
        f"- Average tables: {aggregate.get('average_tables', 0)}",
        f"- Average figures: {aggregate.get('average_figures', 0)}",
        "",
        "## Bottlenecks",
        "",
    ]
    bottlenecks = summary.get("bottlenecks") if isinstance(summary.get("bottlenecks"), list) else []
    lines.extend(
        f"- {item['category']}: {item['count']} case(s), route {item['recommended_phase']}"
        for item in bottlenecks
    )
    if not bottlenecks:
        lines.append("- None")
    lines.extend(["", "## Root Cause Clusters", ""])
    clusters = (
        summary.get("root_cause_clusters")
        if isinstance(summary.get("root_cause_clusters"), list)
        else []
    )
    lines.extend(
        f"- P{item['priority']} {item['cluster']}: {item['case_count']} case(s), "
        f"route {item['recommended_phase']}. {item['action']}"
        for item in clusters
    )
    if not clusters:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Case Findings",
            "",
            "| Case | Contest | Drill | Categories | Failed gates | Next phase |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for item in summary.get("case_findings", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('case_id', '')} | "
            f"{float(item.get('contest_score', 0)):.2f} | "
            f"{float(item.get('drill_score', 0)):.2f} | "
            f"{', '.join(item.get('failure_categories', [])) or 'none'} | "
            f"{', '.join(item.get('failed_gates', [])) or 'none'} | "
            f"{item.get('recommended_phase', '')} |"
        )
    return "\n".join(lines)


def run_pressure_audit(
    drill_path: Path,
    output_dir: Path,
    *,
    contest_path: Path | None = None,
    time_budget_seconds: int | None = None,
    contest_score_floor: float = 85.0,
) -> dict[str, Any]:
    drill_summary = _read_json(drill_path)
    contest_summary = _read_json(contest_path) if contest_path else None
    summary = build_pressure_audit(
        drill_summary,
        contest_summary,
        time_budget_seconds=time_budget_seconds,
        contest_score_floor=contest_score_floor,
    )
    write_pressure_audit(summary, output_dir)
    return summary


def _case_finding(
    drill: dict[str, Any],
    contest: dict[str, Any] | None,
    *,
    time_budget_seconds: int | None,
    contest_score_floor: float,
) -> dict[str, Any]:
    gates = drill.get("quality_gates") if isinstance(drill.get("quality_gates"), dict) else {}
    artifacts = drill.get("artifacts") if isinstance(drill.get("artifacts"), dict) else {}
    categories: list[str] = []
    reasons: list[str] = []

    failed_gates = [
        key for key, value in gates.items() if str(value).lower() in {"failed", "blocked", "fail"}
    ]
    missing_gates = [key for key in EXPECTED_GATES if key not in gates]
    missing_artifacts = [key for key in REQUIRED_ARTIFACTS if key not in artifacts]

    if drill.get("execution_status") != "success":
        _add(categories, "execution_failure")
        reasons.append("execution did not finish successfully")
    if int(drill.get("error_count") or 0) > 0:
        _add(categories, "workflow_errors")
        reasons.append("workflow recorded errors")
    if failed_gates:
        _add(categories, "gate_failure")
        reasons.append("failed gates: " + ", ".join(failed_gates))
    if missing_gates:
        _add(categories, "gate_missing")
        reasons.append("missing gate evaluations: " + ", ".join(missing_gates))
    if missing_artifacts:
        _add(categories, "artifact_gap")
        reasons.append("missing artifacts: " + ", ".join(missing_artifacts))
    if drill.get("missing_models"):
        _add(categories, "model_output_gap")
        reasons.append("missing models: " + ", ".join(map(str, drill.get("missing_models", [])[:5])))
    if int(drill.get("table_count") or 0) < AWARD_TABLE_TARGET:
        _add(categories, "sparse_tables")
    if int(drill.get("figure_count") or 0) < AWARD_FIGURE_TARGET:
        _add(categories, "sparse_figures")
    if float(drill.get("paper_quality_score") or 0.0) < AWARD_PAPER_TARGET:
        _add(categories, "weak_paper_quality")
    if time_budget_seconds and float(drill.get("elapsed_seconds") or 0.0) > 0.85 * time_budget_seconds:
        _add(categories, "slow_run")

    contest_score = float((contest or {}).get("contest_score") or 0.0)
    if contest is not None and contest_score < contest_score_floor:
        _add(categories, "low_contest_score")
    correctness = (contest or {}).get("answer_correctness_audit")
    if isinstance(correctness, dict) and correctness.get("applicable") and not correctness.get("passed"):
        _add(categories, "answer_correctness_failed")
        reasons.append("answer correctness expectations failed")

    return {
        "case_id": str(drill.get("case_id", "")),
        "workspace": str(drill.get("workspace", "")),
        "drill_score": float(drill.get("score") or 0.0),
        "contest_score": contest_score,
        "readiness_band": str((contest or {}).get("readiness_band", "")),
        "failure_categories": categories,
        "failed_gates": failed_gates,
        "missing_gates": missing_gates,
        "missing_artifacts": missing_artifacts,
        "recommended_phase": _recommended_phase(categories, failed_gates),
        "reasons": reasons,
    }


def _rank_bottlenecks(
    category_counts: Counter[str],
    gate_counts: Counter[str],
) -> list[dict[str, Any]]:
    items = [
        {
            "category": category,
            "count": count,
            "recommended_phase": _recommended_phase([category], []),
        }
        for category, count in category_counts.items()
        if count > 0
    ]
    items.extend(
        {
            "category": f"gate:{gate}",
            "count": count,
            "recommended_phase": _recommended_phase(["gate_failure"], [gate]),
        }
        for gate, count in gate_counts.items()
        if count > 0
    )
    return sorted(items, key=lambda item: (-int(item["count"]), str(item["category"])))


def _cluster_root_causes(case_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for rule in ROOT_CAUSE_RULES:
        matched_cases: list[str] = []
        matched_categories: Counter[str] = Counter()
        matched_gates: Counter[str] = Counter()
        for finding in case_findings:
            categories = set(finding.get("failure_categories", []))
            failed_gates = set(finding.get("failed_gates", []))
            category_hits = categories & set(rule["categories"])
            gate_hits = failed_gates & set(rule["gates"])
            if not category_hits and not gate_hits:
                continue
            matched_cases.append(str(finding.get("case_id", "")))
            matched_categories.update(category_hits)
            matched_gates.update(gate_hits)
        if not matched_cases:
            continue
        case_count = len(matched_cases)
        severity = int(rule["severity"])
        priority_score = severity * 10 + case_count * 3 + len(matched_categories) + len(matched_gates)
        clusters.append(
            {
                "cluster": str(rule["cluster"]),
                "case_count": case_count,
                "priority_score": priority_score,
                "priority": _priority_band(priority_score),
                "recommended_phase": str(rule["recommended_phase"]),
                "action": str(rule["action"]),
                "categories": dict(sorted(matched_categories.items())),
                "gates": dict(sorted(matched_gates.items())),
                "sample_cases": matched_cases[:5],
            }
        )
    return sorted(
        clusters,
        key=lambda item: (-int(item["priority_score"]), str(item["cluster"])),
    )


def _priority_band(score: int) -> int:
    if score >= 48:
        return 0
    if score >= 38:
        return 1
    if score >= 28:
        return 2
    return 3


def _recommended_phase(categories: list[str], failed_gates: list[str]) -> str:
    category_set = set(categories)
    gate_set = set(failed_gates)
    if category_set & {"execution_failure", "model_output_gap"}:
        return "code_plan"
    if "answer_correctness_failed" in category_set:
        return "result_analysis"
    if "strong_baseline_gate" in gate_set or "innovation_evidence_gate" in gate_set:
        return "experiment_plan"
    if "task_traceability_gate" in gate_set:
        return "evidence_mapping"
    if category_set & {
        "weak_paper_quality",
        "sparse_tables",
        "sparse_figures",
        "gate_failure",
        "low_contest_score",
    }:
        return "section_writing"
    if category_set & {"artifact_gap", "gate_missing"}:
        return "export"
    return "none"


def _contest_results_by_case(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    return {
        str(item.get("case_id", "")): item
        for item in summary.get("results", [])
        if isinstance(item, dict)
    }


def _gate_passed(gates: Any, key: str) -> bool:
    return isinstance(gates, dict) and str(gates.get(key, "")).lower() == "passed"


def _add(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _average(values: Any) -> float:
    parsed = [float(value) for value in values if _is_number(value)]
    if not parsed:
        return 0.0
    return round(sum(parsed) / len(parsed), 2)


def _rate(values: Any) -> float:
    parsed = [bool(value) for value in values]
    if not parsed:
        return 0.0
    return round(sum(parsed) / len(parsed), 4)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a P3 end-to-end pressure audit.")
    parser.add_argument("--drill", type=Path, default=Path("benchmarks/results/real_case_drill.json"))
    parser.add_argument("--contest", type=Path, default=Path("benchmarks/results/contest_simulation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--time-budget-seconds", type=int, default=None)
    parser.add_argument("--contest-score-floor", type=float, default=85.0)
    args = parser.parse_args()
    contest_path = args.contest if args.contest.exists() else None
    summary = run_pressure_audit(
        args.drill,
        args.output_dir,
        contest_path=contest_path,
        time_budget_seconds=args.time_budget_seconds,
        contest_score_floor=args.contest_score_floor,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "case_findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
