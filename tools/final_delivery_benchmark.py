from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.file_tool import write_text


DEFAULT_THRESHOLDS = {
    "min_average_contest_score": 88.0,
    "min_average_blind_review_score": 82.0,
    "min_first_prize_ready_rate": 0.2,
    "max_high_risk_case_rate": 0.25,
    "min_export_ready_rate": 0.8,
    "min_paper_evidence_pass_rate": 0.8,
    "max_p0_cluster_count": 0,
    "min_candidate_profile": 1.0,
}


def build_final_delivery_benchmark(
    contest_summary: dict[str, Any],
    pressure_audit: dict[str, Any] | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    case_count = int(contest_summary.get("case_count") or 0)
    high_risk_count = int(contest_summary.get("high_risk_case_count") or 0)
    high_risk_rate = high_risk_count / max(case_count, 1)
    pressure_aggregate = pressure_audit.get("aggregate", {}) if isinstance(pressure_audit, dict) else {}
    root_clusters = (
        pressure_audit.get("root_cause_clusters", [])
        if isinstance(pressure_audit, dict) and isinstance(pressure_audit.get("root_cause_clusters"), list)
        else []
    )
    p0_clusters = [item for item in root_clusters if int(item.get("priority", 9)) == 0]
    blocking_findings = _blocking_findings(contest_summary, pressure_audit)
    checks = {
        "candidate_profile": _check_min(
            1.0 if _is_candidate_profile(contest_summary) else 0.0,
            limits["min_candidate_profile"],
        ),
        "average_contest_score": _check_min(
            contest_summary.get("average_contest_score"),
            limits["min_average_contest_score"],
        ),
        "average_blind_review_score": _check_min(
            contest_summary.get("average_blind_review_score"),
            limits["min_average_blind_review_score"],
        ),
        "first_prize_ready_rate": _check_min(
            contest_summary.get("first_prize_ready_rate"),
            limits["min_first_prize_ready_rate"],
        ),
        "high_risk_case_rate": _check_max(high_risk_rate, limits["max_high_risk_case_rate"]),
        "export_ready_rate": _check_min(
            pressure_aggregate.get("export_ready_rate"),
            limits["min_export_ready_rate"],
        ),
        "paper_evidence_pass_rate": _check_min(
            pressure_aggregate.get("paper_evidence_pass_rate"),
            limits["min_paper_evidence_pass_rate"],
        ),
        "p0_root_cause_clusters": _check_max(
            len(p0_clusters),
            limits["max_p0_cluster_count"],
        ),
    }
    failed_checks = [key for key, item in checks.items() if not item["passed"]]
    status = _delivery_status(failed_checks, blocking_findings, contest_summary)
    return {
        "schema_version": "1.0",
        "case_count": case_count,
        "status": status,
        "passed": status in {"award_candidate", "manual_review_required"},
        "thresholds": limits,
        "checks": checks,
        "failed_checks": failed_checks,
        "blocking_findings": blocking_findings,
        "p0_root_cause_clusters": p0_clusters,
        "recommendations": _recommendations(failed_checks, blocking_findings, p0_clusters),
    }


def write_final_delivery_benchmark(summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_text(
        output_dir / "final_delivery_benchmark.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    md_path = write_text(output_dir / "final_delivery_benchmark.md", format_final_delivery_benchmark(summary))
    return {"json": json_path, "markdown": md_path}


def format_final_delivery_benchmark(summary: dict[str, Any]) -> str:
    lines = [
        "# Final Delivery Benchmark",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Passed: {summary.get('passed', False)}",
        "",
        "## Checks",
        "",
        "| Check | Value | Threshold | Passed |",
        "|---|---:|---:|:---:|",
    ]
    checks = summary.get("checks") if isinstance(summary.get("checks"), dict) else {}
    for key, item in checks.items():
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {key} | {float(item.get('value', 0)):.2f} | "
            f"{float(item.get('threshold', 0)):.2f} | "
            f"{'yes' if item.get('passed') else 'no'} |"
        )
    lines.extend(["", "## Blocking Findings", ""])
    blockers = summary.get("blocking_findings") if isinstance(summary.get("blocking_findings"), list) else []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Recommendations", ""])
    recommendations = summary.get("recommendations") if isinstance(summary.get("recommendations"), list) else []
    lines.extend(f"- {item}" for item in recommendations) if recommendations else lines.append("- None")
    return "\n".join(lines)


def run_final_delivery_benchmark(
    contest_path: Path,
    output_dir: Path,
    *,
    pressure_path: Path | None = None,
) -> dict[str, Any]:
    contest_summary = _read_json(contest_path)
    pressure_audit = _read_json(pressure_path) if pressure_path else {}
    summary = build_final_delivery_benchmark(contest_summary, pressure_audit)
    write_final_delivery_benchmark(summary, output_dir)
    return summary


def _check_min(value: Any, threshold: float) -> dict[str, Any]:
    parsed = _float(value)
    return {
        "value": parsed,
        "threshold": float(threshold),
        "operator": ">=",
        "passed": parsed >= float(threshold),
    }


def _check_max(value: Any, threshold: float) -> dict[str, Any]:
    parsed = _float(value)
    return {
        "value": parsed,
        "threshold": float(threshold),
        "operator": "<=",
        "passed": parsed <= float(threshold),
    }


def _blocking_findings(
    contest_summary: dict[str, Any],
    pressure_audit: dict[str, Any] | None,
) -> list[str]:
    findings: list[str] = []
    for item in contest_summary.get("results", []):
        if not isinstance(item, dict):
            continue
        if item.get("readiness_band") == "not_competitive":
            findings.append(f"{item.get('case_id', '')}: not_competitive")
        risks = item.get("risks") if isinstance(item.get("risks"), list) else []
        if any("answer correctness" in str(risk).lower() for risk in risks):
            findings.append(f"{item.get('case_id', '')}: answer correctness failed")
    if isinstance(pressure_audit, dict):
        for item in pressure_audit.get("case_findings", []):
            if not isinstance(item, dict):
                continue
            failed_gates = item.get("failed_gates") if isinstance(item.get("failed_gates"), list) else []
            if failed_gates:
                findings.append(f"{item.get('case_id', '')}: failed gates " + ", ".join(failed_gates[:4]))
    return _dedupe(findings)


def _delivery_status(
    failed_checks: list[str],
    blocking_findings: list[str],
    contest_summary: dict[str, Any],
) -> str:
    first_prize_rate = _float(contest_summary.get("first_prize_ready_rate"))
    if not failed_checks and not blocking_findings and first_prize_rate >= 0.6:
        return "award_candidate"
    if {
        "candidate_profile",
        "p0_root_cause_clusters",
        "average_contest_score",
        "average_blind_review_score",
    } & set(failed_checks):
        return "not_ready"
    if blocking_findings:
        return "not_ready"
    return "manual_review_required"


def _recommendations(
    failed_checks: list[str],
    blocking_findings: list[str],
    p0_clusters: list[Any],
) -> list[str]:
    recommendations: list[str] = []
    if "candidate_profile" in failed_checks:
        recommendations.append("Rerun contest simulation with --candidate-profile before final delivery judgment.")
    if p0_clusters:
        recommendations.append("Resolve P0 root-cause clusters before another final delivery run.")
    if "average_contest_score" in failed_checks or "average_blind_review_score" in failed_checks:
        recommendations.append("Run multi-case P3 pressure tests and improve the lowest scoring dimensions first.")
    if "export_ready_rate" in failed_checks or "paper_evidence_pass_rate" in failed_checks:
        recommendations.append("Run formal export and paper evidence gates for every candidate case.")
    if "high_risk_case_rate" in failed_checks:
        recommendations.append("Reduce risky/not_competitive cases by applying root-cause rework plans.")
    if blocking_findings:
        recommendations.append("Clear per-case blocking findings before treating the output as submit-ready.")
    return _dedupe(recommendations)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_candidate_profile(contest_summary: dict[str, Any]) -> bool:
    profile = contest_summary.get("run_profile")
    return isinstance(profile, dict) and profile.get("candidate_profile") is True


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final award-candidate delivery benchmark.")
    parser.add_argument("--contest", type=Path, default=Path("benchmarks/results/contest_simulation.json"))
    parser.add_argument("--pressure", type=Path, default=Path("benchmarks/results/pressure_audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    pressure_path = args.pressure if args.pressure.exists() else None
    summary = run_final_delivery_benchmark(args.contest, args.output_dir, pressure_path=pressure_path)
    print(json.dumps({key: value for key, value in summary.items() if key != "checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
