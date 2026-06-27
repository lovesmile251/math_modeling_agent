from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from agents.base import ClaimEvidenceMap
from tools.file_tool import write_text


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
_CLAIM_ID = re.compile(r"\bC-\d{3,}\b")
_RESULT_TERMS = (
    "结果", "表明", "达到", "提高", "降低", "排名", "误差", "准确",
    "均值", "最优", "预测", "权重", "得分", "概率", "效率", "相关",
    "result", "achieve", "increase", "decrease", "error", "accuracy",
    "mean", "optimal", "forecast", "weight", "score", "probability",
)


@dataclass(frozen=True)
class NumericClaimCheck:
    line_number: int
    text: str
    values: list[str]
    evidence_ids: list[str] = field(default_factory=list)
    matched_sources: list[str] = field(default_factory=list)
    mapped: bool = False


@dataclass(frozen=True)
class TraceabilityReport:
    passed: bool
    coverage_pct: float
    total_numeric_claims: int
    mapped_numeric_claims: int
    threshold_pct: float
    claims: list[NumericClaimCheck] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_numeric_traceability(
    paper_text: str,
    workspace,
    claim_map: ClaimEvidenceMap | None,
    threshold_pct: float = 70.0,
) -> TraceabilityReport:
    evidence_ids = {claim.claim_id: claim for claim in (claim_map.claims if claim_map else [])}
    source_values = _load_source_values(workspace.tables_dir)
    checks: list[NumericClaimCheck] = []

    for line_number, line in enumerate(paper_text.splitlines(), start=1):
        values = _substantive_values(line)
        if not values:
            continue
        cited_ids = [item for item in _CLAIM_ID.findall(line) if item in evidence_ids]
        matched_sources: set[str] = set()
        mapped_values = 0
        for raw in values:
            numeric = _to_float(raw)
            if numeric is None:
                continue
            matches = _matching_sources(numeric, source_values)
            if matches:
                mapped_values += 1
                matched_sources.update(matches)
                continue
            if any(_claim_contains_value(evidence_ids[item], numeric) for item in cited_ids):
                mapped_values += 1
        checks.append(
            NumericClaimCheck(
                line_number=line_number,
                text=line.strip()[:500],
                values=values,
                evidence_ids=cited_ids,
                matched_sources=sorted(matched_sources)[:10],
                mapped=mapped_values == len(values),
            )
        )

    total = len(checks)
    mapped = sum(1 for check in checks if check.mapped)
    coverage = 100.0 if total == 0 else round(mapped / total * 100, 1)
    issues = [
        f"line {check.line_number}: unmapped numeric claim: {check.text}"
        for check in checks
        if not check.mapped
    ]
    return TraceabilityReport(
        passed=coverage >= threshold_pct,
        coverage_pct=coverage,
        total_numeric_claims=total,
        mapped_numeric_claims=mapped,
        threshold_pct=threshold_pct,
        claims=checks,
        issues=issues,
    )


def write_traceability_report(workspace, report: TraceabilityReport) -> Path:
    return write_text(
        workspace.logs_dir / "traceability_report.json",
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
    )


def _substantive_values(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("```"):
        return []
    if _looks_like_markdown_table_header(stripped):
        return []
    if stripped.startswith("[") and re.match(r"^\[\d+\]", stripped):
        return []
    if not any(term.lower() in stripped.lower() for term in _RESULT_TERMS):
        return []
    stripped = _CLAIM_ID.sub("", stripped)
    values: list[str] = []
    for raw in _NUMBER.findall(stripped):
        numeric = _to_float(raw)
        if numeric is None:
            continue
        if 1900 <= numeric <= 2100 and raw.isdigit():
            continue
        values.append(raw)
    return values


def _load_source_values(tables_dir: Path) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    for path in sorted(tables_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
        numeric_columns = _numeric_columns(frame)
        numeric = pd.DataFrame({column: numeric_columns[column] for column in numeric_columns})
        for value in numeric.to_numpy().ravel():
            if pd.notna(value) and math.isfinite(float(value)):
                values.append((float(value), path.name))
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            for value in (
                series.min(),
                series.max(),
                series.mean(),
                series.std(),
                series.median(),
                series.quantile(0.25),
                series.quantile(0.75),
            ):
                if pd.notna(value) and math.isfinite(float(value)):
                    values.append((float(value), path.name))
    return values


def _numeric_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    numeric: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if series.notna().any():
            numeric[str(column)] = series
    return numeric


def _looks_like_markdown_table_header(line: str) -> bool:
    if not line.startswith("|"):
        return False
    cells = [cell.strip().lower() for cell in line.strip("|").split("|")]
    if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
        return True
    header_tokens = {
        "unnamed: 0",
        "count",
        "unique",
        "top",
        "freq",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    }
    matched = sum(cell in header_tokens for cell in cells)
    return matched >= max(3, len(cells) // 2)


def _matching_sources(value: float, source_values: list[tuple[float, str]]) -> set[str]:
    tolerance = max(1e-6, abs(value) * 5e-4)
    return {
        source
        for candidate, source in source_values
        if abs(candidate - value) <= tolerance
    }


def _claim_contains_value(claim, value: float) -> bool:
    text = f"{claim.claim} {claim.calculation}"
    for raw in _NUMBER.findall(text):
        candidate = _to_float(raw)
        if candidate is not None and abs(candidate - value) <= max(1e-6, abs(value) * 5e-4):
            return True
    return False


def _to_float(raw: str) -> float | None:
    try:
        value = float(raw.rstrip("%"))
    except ValueError:
        return None
    return value / 100 if raw.endswith("%") else value
