from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from tools.file_tool import write_text


def load_gold_expectations(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load case-level answer expectations from JSON.

    Supported top-level formats:
    - [{"case_id": "...", ...}]
    - {"cases": [{"case_id": "...", ...}]}
    - {"gold": [{"case_id": "...", ...}]}
    """

    if path is None or not Path(path).exists():
        return {}
    payload = _read_json(Path(path))
    if isinstance(payload, dict):
        items = payload.get("cases", payload.get("gold", []))
    else:
        items = payload
    if not isinstance(items, list):
        return {}

    expectations: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("case_id"):
            continue
        if has_answer_expectations(item):
            expectations[str(item["case_id"])] = item
    return expectations


def has_answer_expectations(expectation: dict[str, Any] | None) -> bool:
    if not expectation:
        return False
    return bool(
        expectation.get("expected_numeric_ranges")
        or expectation.get("numeric_ranges")
        or expectation.get("expected_decisions")
        or expectation.get("decisions")
    )


def audit_workspace_correctness(
    workspace: Path,
    *,
    case_id: str | None = None,
    expectation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    if expectation is None and case_id:
        expectation_path = workspace / "logs" / "answer_expectations.json"
        loaded = load_gold_expectations(expectation_path)
        expectation = loaded.get(case_id)
    registry = _read_json(workspace / "logs" / "result_registry.json")
    registry_entries = registry.get("entries", []) if isinstance(registry, dict) else []
    table_entries = [
        entry
        for entry in registry_entries
        if isinstance(entry, dict) and entry.get("type") == "table"
    ]
    paper_text = _read_text(workspace / "paper" / "paper_draft.md")
    return audit_answer_correctness(
        table_entries=table_entries,
        paper_text=paper_text,
        expectation=expectation,
    )


def audit_answer_correctness(
    *,
    table_entries: list[dict[str, Any]],
    paper_text: str,
    expectation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check generated answers against case-specific numeric and decision expectations."""

    if not has_answer_expectations(expectation):
        return {
            "applicable": False,
            "passed": True,
            "pass_rate": 1.0,
            "numeric_checks": [],
            "decision_checks": [],
            "failures": [],
        }

    assert expectation is not None
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
    passed_count = sum(1 for item in checks if item["passed"])
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


def summarize_correctness(audits: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [item for item in audits if item.get("applicable")]
    return {
        "case_count": len(audits),
        "applicable_case_count": len(applicable),
        "pass_rate": _average([float(item.get("pass_rate", 0.0)) for item in applicable]),
        "passed_case_count": sum(bool(item.get("passed")) for item in applicable),
        "results": audits,
    }


def write_correctness_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        output_dir / "answer_correctness_audit.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    write_text(output_dir / "answer_correctness_audit.md", format_correctness_report(summary))


def format_correctness_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Answer Correctness Audit",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Applicable cases: {summary['applicable_case_count']}",
        f"- Average pass rate: {summary['pass_rate']:.1%}",
        f"- Passed cases: {summary['passed_case_count']}",
        "",
        "| Case | Applicable | Passed | Pass rate | Failures |",
        "|---|:---:|:---:|---:|---|",
    ]
    for item in summary.get("results", []):
        failures = "; ".join(
            f"{failure.get('label')}: {failure.get('reason')}"
            for failure in item.get("failures", [])[:3]
        )
        lines.append(
            f"| {item.get('case_id', '')} | {bool(item.get('applicable'))} | "
            f"{bool(item.get('passed'))} | {float(item.get('pass_rate', 0.0)):.1%} | "
            f"{failures or 'None'} |"
        )
    return "\n".join(lines)


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


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated answers against gold expectations.")
    parser.add_argument("--workspace", type=Path, action="append", required=True)
    parser.add_argument("--gold-expectations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    expectations = load_gold_expectations(args.gold_expectations)
    audits = []
    for workspace in args.workspace:
        case_id = workspace.name
        audit = audit_workspace_correctness(
            workspace,
            case_id=case_id,
            expectation=expectations.get(case_id),
        )
        audit["case_id"] = case_id
        audit["workspace"] = str(workspace)
        audits.append(audit)
    summary = summarize_correctness(audits)
    write_correctness_summary(summary, args.output_dir)
    compact = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
