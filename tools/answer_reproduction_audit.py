from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from tools.answer_correctness import (
    audit_answer_correctness as _shared_audit_answer_correctness,
    load_gold_expectations,
)
from tools.file_tool import write_text
from tools.paper_quality import (
    AWARD_MEDIAN_CHARS,
    AWARD_MEDIAN_EQUATIONS,
    AWARD_MEDIAN_FIGURES,
    AWARD_MEDIAN_REFERENCES,
    AWARD_MEDIAN_TABLES,
    evaluate_paper_quality,
)


MEAN_STD_RE = re.compile(
    r"mean\((?P<column>.+?)\)\s*=\s*(?P<mean>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?),\s*"
    r"std\(.+?\)\s*=\s*(?P<std>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    flags=re.IGNORECASE,
)
REVIEW_MODEL_COUNT_RE = re.compile(r"成功产出结果的模型数[:：]\s*(\d+)")


def audit_from_simulation_report(
    simulation_report: Path,
    *,
    output_dir: Path,
    max_claims_per_case: int = 80,
    gold_expectations_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(simulation_report.read_text(encoding="utf-8"))
    case_items = [item for item in payload.get("results", []) if isinstance(item, dict)]
    gold_expectations = load_gold_expectations(gold_expectations_path)
    audits = [
        audit_workspace(
            Path(str(item.get("workspace", ""))),
            case_id=str(item.get("case_id", "")),
            max_claims=max_claims_per_case,
            gold_expectation=gold_expectations.get(str(item.get("case_id", ""))),
        )
        for item in case_items
    ]
    summary = summarize_audits(audits)
    write_audit_summary(summary, output_dir)
    return summary


def audit_workspace(
    workspace: Path,
    *,
    case_id: str | None = None,
    max_claims: int = 80,
    gold_expectation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    case_id = case_id or workspace.name
    registry = _read_json(workspace / "logs" / "result_registry.json")
    claim_map = _read_json(workspace / "logs" / "claim_evidence_map.json")
    traceability = _read_json(workspace / "logs" / "traceability_report.json")
    run_summary = _read_json(workspace / "logs" / "run_summary.json")
    review_text = _read_text(workspace / "paper" / "review_report.md")
    paper_text = _read_text(workspace / "paper" / "paper_draft.md")

    registry_entries = registry.get("entries", []) if isinstance(registry, dict) else []
    table_entries = [entry for entry in registry_entries if isinstance(entry, dict) and entry.get("type") == "table"]
    figure_entries = [entry for entry in registry_entries if isinstance(entry, dict) and entry.get("type") == "figure"]
    hash_audit = _verify_registry_hashes(table_entries)

    claims = claim_map.get("claims", []) if isinstance(claim_map, dict) else []
    claim_audit = _verify_claims(claims, max_claims=max_claims)
    model_count_audit = _audit_model_count(run_summary, review_text)
    paper_audit = _audit_paper_against_award_style(paper_text, workspace)
    correctness_audit = _shared_audit_answer_correctness(
        table_entries=table_entries,
        paper_text=paper_text,
        expectation=gold_expectation,
    )
    traceability_coverage = float(traceability.get("coverage_pct", 0.0)) if isinstance(traceability, dict) else 0.0
    traceability_passed = bool(traceability.get("passed", False)) if isinstance(traceability, dict) else False

    risks = _collect_risks(
        claim_audit=claim_audit,
        hash_audit=hash_audit,
        model_count_audit=model_count_audit,
        paper_audit=paper_audit,
        traceability_coverage=traceability_coverage,
        traceability_passed=traceability_passed,
        correctness_audit=correctness_audit,
    )
    score = _audit_score(
        claim_audit=claim_audit,
        hash_audit=hash_audit,
        model_count_audit=model_count_audit,
        paper_audit=paper_audit,
        traceability_coverage=traceability_coverage,
        traceability_passed=traceability_passed,
        correctness_audit=correctness_audit,
    )
    return {
        "case_id": case_id,
        "workspace": str(workspace),
        "audit_score": score,
        "reproducibility_band": _band(score, risks),
        "numeric_claim_audit": claim_audit,
        "registry_hash_audit": hash_audit,
        "model_count_audit": model_count_audit,
        "traceability": {
            "passed": traceability_passed,
            "coverage_pct": traceability_coverage,
        },
        "answer_correctness_audit": correctness_audit,
        "award_style_audit": paper_audit,
        "artifact_counts": {
            "registered_tables": len(table_entries),
            "registered_figures": len(figure_entries),
            "claim_count": len(claims),
        },
        "risks": risks,
    }


def summarize_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(audits),
        "average_audit_score": _average([item["audit_score"] for item in audits]),
        "verified_claim_rate": _average([
            item["numeric_claim_audit"]["verified_rate"]
            for item in audits
        ]),
        "hash_pass_rate": _average([
            item["registry_hash_audit"]["hash_pass_rate"]
            for item in audits
        ]),
        "award_alignment_rate": _average([
            item["award_style_audit"]["alignment_rate"]
            for item in audits
        ]),
        "answer_correctness_pass_rate": _average([
            item["answer_correctness_audit"]["pass_rate"]
            for item in audits
            if item.get("answer_correctness_audit", {}).get("applicable")
        ]),
        "answer_correctness_case_count": sum(
            item.get("answer_correctness_audit", {}).get("applicable", False)
            for item in audits
        ),
        "high_risk_case_count": sum(item["reproducibility_band"] == "high_risk" for item in audits),
        "results": audits,
    }


def write_audit_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "answer_reproduction_audit.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    write_text(output_dir / "answer_reproduction_audit.md", format_audit_report(summary))


def format_audit_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 数值答案复现与优秀论文对照审计",
        "",
        f"- 案例数：{summary['case_count']}",
        f"- 平均审计分：{summary['average_audit_score']}",
        f"- 数值声明复现率：{summary['verified_claim_rate']:.1%}",
        f"- 结果表哈希通过率：{summary['hash_pass_rate']:.1%}",
        f"- 优秀论文风格对齐率：{summary['award_alignment_rate']:.1%}",
        f"- 高风险案例数：{summary['high_risk_case_count']}",
        "",
        "## 分题审计",
        "",
        "| 题目 | 审计分 | 分层 | 数值复现率 | 哈希通过率 | 风格对齐率 | 主要风险 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for item in summary["results"]:
        risks = "；".join(item["risks"][:3]) if item["risks"] else "无"
        lines.append(
            f"| {item['case_id']} | {item['audit_score']:.2f} | {item['reproducibility_band']} | "
            f"{item['numeric_claim_audit']['verified_rate']:.1%} | "
            f"{item['registry_hash_audit']['hash_pass_rate']:.1%} | "
            f"{item['award_style_audit']['alignment_rate']:.1%} | {risks} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "该审计复算 evidence map 中可解析的数值声明，校验结果表哈希、论文证据追溯、审稿报告与运行摘要的一致性，并将论文指标与优秀论文中位参考进行对照。它不能证明答案一定正确，但能降低“结果不可复现、证据链断裂、论文与产物矛盾”的风险。",
        ]
    )
    return "\n".join(lines)


def format_audit_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Answer Reproduction and Award-Style Audit",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Average audit score: {summary['average_audit_score']}",
        f"- Numeric claim verified rate: {summary['verified_claim_rate']:.1%}",
        f"- Result table hash pass rate: {summary['hash_pass_rate']:.1%}",
        f"- Award-style alignment rate: {summary['award_alignment_rate']:.1%}",
        f"- Answer correctness applicable cases: {summary['answer_correctness_case_count']}",
        f"- Answer correctness pass rate: {summary['answer_correctness_pass_rate']:.1%}",
        f"- High-risk cases: {summary['high_risk_case_count']}",
        "",
        "## Case Audits",
        "",
        "| Case | Audit score | Band | Numeric reproducibility | Hash pass | Style alignment | Correctness | Main risks |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["results"]:
        risks = "; ".join(item["risks"][:3]) if item["risks"] else "None"
        lines.append(
            f"| {item['case_id']} | {item['audit_score']:.2f} | {item['reproducibility_band']} | "
            f"{item['numeric_claim_audit']['verified_rate']:.1%} | "
            f"{item['registry_hash_audit']['hash_pass_rate']:.1%} | "
            f"{item['award_style_audit']['alignment_rate']:.1%} | "
            f"{item['answer_correctness_audit']['pass_rate']:.1%} | {risks} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This audit recomputes verifiable numeric claims, validates result table hashes, checks traceability and award-style paper density, and optionally checks answers against hidden gold expectations. It reduces reproducibility and evidence-chain risk, but it does not replace human contest judging.",
        ]
    )
    return "\n".join(lines)


def _verify_claims(claims: list[Any], *, max_claims: int) -> dict[str, Any]:
    verifiable = 0
    verified = 0
    failed: list[dict[str, Any]] = []
    source_missing = 0
    for claim in claims[:max_claims]:
        if not isinstance(claim, dict):
            continue
        calculation = str(claim.get("calculation", ""))
        match = MEAN_STD_RE.search(calculation)
        if not match:
            continue
        verifiable += 1
        source_file = _resolve_path(str(claim.get("source_file", "")))
        if not source_file.exists():
            source_missing += 1
            failed.append({"claim_id": claim.get("claim_id"), "reason": "source_file_missing"})
            continue
        column = match.group("column")
        try:
            expected_mean = float(match.group("mean"))
            expected_std = float(match.group("std"))
            df = pd.read_csv(source_file)
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            actual_mean = float(series.mean())
            actual_std = float(series.std())
        except (OSError, KeyError, ValueError, pd.errors.ParserError):
            failed.append({"claim_id": claim.get("claim_id"), "reason": "recompute_failed"})
            continue
        if _close(actual_mean, expected_mean) and _close(actual_std, expected_std):
            verified += 1
        else:
            failed.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "reason": "numeric_mismatch",
                    "expected_mean": expected_mean,
                    "actual_mean": round(actual_mean, 6),
                    "expected_std": expected_std,
                    "actual_std": round(actual_std, 6),
                }
            )
    return {
        "sampled_claims": min(len(claims), max_claims),
        "verifiable_claims": verifiable,
        "verified_claims": verified,
        "verified_rate": round(verified / verifiable, 4) if verifiable else 0.0,
        "source_missing": source_missing,
        "failed_claims": failed[:10],
    }


def _verify_registry_hashes(table_entries: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    passed = 0
    failed: list[str] = []
    for entry in table_entries:
        expected = str(entry.get("sha256", ""))
        path = _resolve_path(str(entry.get("path", "")))
        if not expected or not path.exists():
            continue
        checked += 1
        if _sha256(path) == expected:
            passed += 1
        else:
            failed.append(str(path))
    return {
        "checked_tables": checked,
        "passed_tables": passed,
        "hash_pass_rate": round(passed / checked, 4) if checked else 0.0,
        "failed_tables": failed[:10],
    }


def _audit_answer_correctness(
    *,
    table_entries: list[dict[str, Any]],
    paper_text: str,
    expectation: dict[str, Any] | None,
) -> dict[str, Any]:
    if not expectation:
        return {
            "applicable": False,
            "passed": True,
            "pass_rate": 1.0,
            "numeric_checks": [],
            "decision_checks": [],
            "failures": [],
        }

    numeric_ranges = _as_list(
        expectation.get("expected_numeric_ranges", expectation.get("numeric_ranges", []))
    )
    decisions = _as_list(expectation.get("expected_decisions", expectation.get("decisions", [])))
    numeric_checks = [
        _audit_numeric_range_expectation(item, table_entries=table_entries, paper_text=paper_text)
        for item in numeric_ranges
        if isinstance(item, dict)
    ]
    table_text = _registered_table_text(table_entries)
    decision_checks = [
        _audit_decision_expectation(item, paper_text=f"{paper_text}\n{table_text}")
        for item in decisions
        if isinstance(item, dict)
    ]
    checks = numeric_checks + decision_checks
    passed_count = sum(item["passed"] for item in checks)
    failures = [
        {"type": item["type"], "label": item["label"], "reason": item["reason"]}
        for item in checks
        if not item["passed"]
    ]
    return {
        "applicable": bool(checks),
        "passed": (not checks) or passed_count == len(checks),
        "pass_rate": round(passed_count / len(checks), 4) if checks else 1.0,
        "numeric_checks": numeric_checks,
        "decision_checks": decision_checks,
        "failures": failures,
    }


def _audit_numeric_range_expectation(
    expectation: dict[str, Any],
    *,
    table_entries: list[dict[str, Any]],
    paper_text: str,
) -> dict[str, Any]:
    label = str(
        expectation.get("label")
        or expectation.get("metric")
        or expectation.get("column")
        or expectation.get("name")
        or "numeric_expectation"
    )
    lower, upper = _numeric_bounds(expectation)
    values = _find_expected_numeric_values(expectation, table_entries, paper_text)
    matching = [
        value
        for value in values
        if (lower is None or value >= lower) and (upper is None or value <= upper)
    ]
    passed = bool(matching) and (lower is not None or upper is not None)
    reason = "matched" if passed else "no value in expected range"
    if lower is None and upper is None:
        reason = "missing numeric bounds"
    return {
        "type": "numeric_range",
        "label": label,
        "passed": passed,
        "range": {"min": lower, "max": upper},
        "matched_values": [round(value, 8) for value in matching[:10]],
        "sampled_values": [round(value, 8) for value in values[:20]],
        "reason": reason,
    }


def _audit_decision_expectation(expectation: dict[str, Any], *, paper_text: str) -> dict[str, Any]:
    label = str(expectation.get("label") or expectation.get("name") or "decision_expectation")
    acceptable = [
        str(item).strip().lower()
        for item in _as_list(
            expectation.get(
                "acceptable_values",
                expectation.get("expected_values", expectation.get("values", [])),
            )
        )
        if str(item).strip()
    ]
    required_terms = [
        str(item).strip().lower()
        for item in _as_list(expectation.get("required_terms", []))
        if str(item).strip()
    ]
    haystack = paper_text.lower()
    value_hit = bool(acceptable) and any(value in haystack for value in acceptable)
    terms_hit = bool(required_terms) and all(term in haystack for term in required_terms)
    passed = value_hit or terms_hit
    reason = "matched" if passed else "missing acceptable decision evidence"
    if not acceptable and not required_terms:
        reason = "missing decision criteria"
    return {
        "type": "decision",
        "label": label,
        "passed": passed,
        "acceptable_values": acceptable,
        "required_terms": required_terms,
        "reason": reason,
    }


def _audit_model_count(run_summary: Any, review_text: str) -> dict[str, Any]:
    actual = 0
    if isinstance(run_summary, list):
        for item in run_summary:
            if not isinstance(item, dict):
                continue
            outputs = item.get("model_outputs")
            if isinstance(outputs, dict):
                actual += len(outputs)
            runs = item.get("model_runs")
            if isinstance(runs, list):
                actual += sum(
                    1
                    for run in runs
                    if isinstance(run, dict)
                    and run.get("status") == "success"
                    and run.get("table")
                )
    reported = None
    match = REVIEW_MODEL_COUNT_RE.search(review_text)
    if match:
        reported = int(match.group(1))
    consistent = reported is None or reported == actual
    return {
        "actual_successful_model_tables": actual,
        "review_reported_model_tables": reported,
        "consistent": consistent,
    }


def _audit_paper_against_award_style(paper_text: str, workspace: Path) -> dict[str, Any]:
    if not paper_text.strip():
        return {
            "alignment_rate": 0.0,
            "metrics": {},
            "passed_checks": [],
            "failed_checks": ["paper_missing"],
        }
    report = evaluate_paper_quality(
        paper_text,
        workspace_root=workspace,
        available_figures=[p.name for p in (workspace / "figures").glob("*.png")],
    )
    metrics = report.metrics
    checks = {
        "chars_at_award_median": metrics.get("chars", 0) >= AWARD_MEDIAN_CHARS,
        "equations_at_award_median": metrics.get("equations", 0) >= AWARD_MEDIAN_EQUATIONS,
        "tables_at_award_median": metrics.get("tables", 0) >= AWARD_MEDIAN_TABLES,
        "figures_at_award_median": metrics.get("figures", 0) >= AWARD_MEDIAN_FIGURES,
        "references_at_award_median": metrics.get("references", 0) >= AWARD_MEDIAN_REFERENCES,
        "paper_quality_at_90": report.score >= 90,
    }
    passed = [key for key, ok in checks.items() if ok]
    failed = [key for key, ok in checks.items() if not ok]
    return {
        "alignment_rate": round(len(passed) / len(checks), 4),
        "quality_score": report.score,
        "metrics": metrics,
        "passed_checks": passed,
        "failed_checks": failed,
    }


def _collect_risks(
    *,
    claim_audit: dict[str, Any],
    hash_audit: dict[str, Any],
    model_count_audit: dict[str, Any],
    paper_audit: dict[str, Any],
    traceability_coverage: float,
    traceability_passed: bool,
    correctness_audit: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    if claim_audit["verifiable_claims"] == 0:
        risks.append("没有可自动复算的数值声明")
    elif claim_audit["verified_rate"] < 0.95:
        risks.append("数值声明复现率不足 95%")
    if hash_audit["checked_tables"] == 0:
        risks.append("没有可校验哈希的结果表")
    elif hash_audit["hash_pass_rate"] < 1.0:
        risks.append("结果表哈希校验存在失败")
    if not model_count_audit["consistent"]:
        risks.append("审稿报告中的模型产出数量与运行摘要不一致")
    if not traceability_passed or traceability_coverage < 85:
        risks.append("论文数值证据追溯覆盖不足 85%")
    if paper_audit["alignment_rate"] < 0.75:
        risks.append("优秀论文风格指标对齐不足 75%")
    if correctness_audit.get("applicable") and not correctness_audit.get("passed"):
        risks.append("answer correctness expectations failed")
    return risks


def _audit_score(
    *,
    claim_audit: dict[str, Any],
    hash_audit: dict[str, Any],
    model_count_audit: dict[str, Any],
    paper_audit: dict[str, Any],
    traceability_coverage: float,
    traceability_passed: bool,
    correctness_audit: dict[str, Any],
) -> float:
    claim_score = 25 * claim_audit["verified_rate"]
    hash_score = 15 * hash_audit["hash_pass_rate"]
    model_consistency_score = 15 if model_count_audit["consistent"] else 0
    trace_score = 20 * (min(traceability_coverage, 100.0) / 100.0) * (1.0 if traceability_passed else 0.7)
    award_score = 25 * paper_audit["alignment_rate"]
    base_score = claim_score + hash_score + model_consistency_score + trace_score + award_score
    if not correctness_audit.get("applicable"):
        return round(base_score, 2)
    correctness_score = 15 * float(correctness_audit.get("pass_rate", 0.0))
    return round(base_score * 0.85 + correctness_score, 2)


def _band(score: float, risks: list[str]) -> str:
    hard_risk = any("不一致" in risk or "哈希" in risk or "复现率不足" in risk for risk in risks)
    if score >= 92 and not hard_risk:
        return "reproducible"
    if score >= 82 and not hard_risk:
        return "mostly_reproducible"
    if score >= 70:
        return "needs_manual_review"
    return "high_risk"


def _load_gold_expectations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    if isinstance(payload, dict):
        items = payload.get("cases", payload.get("gold", []))
    else:
        items = payload
    expectations: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return expectations
    for item in items:
        if not isinstance(item, dict) or "case_id" not in item:
            continue
        if (
            item.get("expected_numeric_ranges")
            or item.get("numeric_ranges")
            or item.get("expected_decisions")
            or item.get("decisions")
        ):
            expectations[str(item["case_id"])] = item
    return expectations


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _numeric_bounds(expectation: dict[str, Any]) -> tuple[float | None, float | None]:
    lower = _optional_float(expectation.get("min", expectation.get("lower")))
    upper = _optional_float(expectation.get("max", expectation.get("upper")))
    target = _optional_float(expectation.get("target", expectation.get("expected")))
    if target is not None and lower is None and upper is None:
        tolerance = _optional_float(expectation.get("tolerance"))
        relative_tolerance = _optional_float(expectation.get("relative_tolerance"))
        if tolerance is None and relative_tolerance is not None:
            tolerance = abs(target) * relative_tolerance
        if tolerance is None:
            tolerance = max(1e-6, abs(target) * 1e-4)
        lower = target - tolerance
        upper = target + tolerance
    return lower, upper


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _find_expected_numeric_values(
    expectation: dict[str, Any],
    table_entries: list[dict[str, Any]],
    paper_text: str,
) -> list[float]:
    values: list[float] = []
    metric = str(expectation.get("metric", expectation.get("label", ""))).strip().lower()
    column = str(expectation.get("column", "")).strip()
    source_names = {str(item).lower() for item in _as_list(expectation.get("source_files", []))}
    for entry in table_entries:
        path = _resolve_path(str(entry.get("path", "")))
        if source_names and path.name.lower() not in source_names and str(path).lower() not in source_names:
            continue
        values.extend(_numeric_values_from_table(path, column=column, metric=metric))
    values.extend(_numeric_values_from_text(paper_text, metric=metric))
    return values


def _numeric_values_from_table(path: Path, *, column: str, metric: str) -> list[float]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return []
    if column and column in df.columns:
        return [
            float(value)
            for value in pd.to_numeric(df[column], errors="coerce").dropna().tolist()
            if math.isfinite(float(value))
        ]
    if metric:
        values: list[float] = []
        for _, row in df.iterrows():
            row_text = " ".join(str(value).lower() for value in row.tolist())
            if metric not in row_text:
                continue
            for value in row.tolist():
                parsed = _optional_float(value)
                if parsed is not None:
                    values.append(parsed)
        return values
    numeric = df.apply(pd.to_numeric, errors="coerce")
    return [
        float(value)
        for value in numeric.to_numpy().ravel().tolist()
        if value == value and math.isfinite(float(value))
    ]


def _numeric_values_from_text(text: str, *, metric: str) -> list[float]:
    search_text = text
    if metric:
        spans = []
        lowered = text.lower()
        start = 0
        while True:
            index = lowered.find(metric, start)
            if index < 0:
                break
            spans.append(text[index : index + 220])
            start = index + len(metric)
        search_text = "\n".join(spans)
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", search_text, flags=re.IGNORECASE):
        parsed = _optional_float(match.group(0).rstrip("%"))
        if parsed is not None:
            values.append(parsed)
    return values


def _registered_table_text(table_entries: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in table_entries:
        path = _resolve_path(str(entry.get("path", "")))
        if not path.exists():
            continue
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore")[:20000])
        except OSError:
            continue
    return "\n".join(parts)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float) -> bool:
    if math.isnan(actual) and math.isnan(expected):
        return True
    return abs(actual - expected) <= max(1e-4, 1e-4 * max(abs(actual), abs(expected), 1.0))


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit numeric answer reproducibility and award-paper alignment.")
    parser.add_argument("--simulation-report", type=Path, default=Path("benchmarks/results/contest_simulation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--max-claims-per-case", type=int, default=80)
    parser.add_argument("--gold-expectations", type=Path, default=None)
    args = parser.parse_args()

    summary = audit_from_simulation_report(
        args.simulation_report,
        output_dir=args.output_dir,
        max_claims_per_case=max(args.max_claims_per_case, 1),
        gold_expectations_path=args.gold_expectations,
    )
    compact = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
