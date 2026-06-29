from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.base import FormulationSpec, ResultRegistry, TaskDeliverableSpec
from tools.file_tool import write_text


_TASK_TYPE_TABLE_TOKENS: dict[str, tuple[str, ...]] = {
    "forecast": ("forecast", "trend", "grey", "smoothing", "error"),
    "evaluation": ("weight", "rank", "score", "topsis", "entropy", "ahp", "vikor"),
    "optimization": ("optimization", "allocation", "plan", "route", "control", "esp"),
    "statistics": ("correlation", "regression", "hypothesis", "anova", "estimate"),
    "classification": ("classification", "classifier", "confusion", "label"),
    "clustering": ("cluster", "profile", "kmeans", "dbscan"),
    "network": ("network", "graph", "path", "community", "centrality", "flow"),
    "simulation": ("simulation", "monte", "sensitivity", "trajectory"),
    "exploration": ("describe", "summary", "profile"),
}


def build_task_traceability_report(
    *,
    deliverables: list[TaskDeliverableSpec],
    formulation: FormulationSpec | None,
    registry: ResultRegistry | None,
    paper_text: str | None = None,
) -> dict[str, Any]:
    """Check task-level closure across model, result table, and paper text."""

    stages = _stages_by_id(formulation)
    table_entries = _table_entries(registry)
    items: list[dict[str, Any]] = []
    issues: list[str] = []

    for index, deliverable in enumerate(deliverables or [], start=1):
        task_id = deliverable.task_id or f"Q{index}"
        task_type = deliverable.task_type or "exploration"
        stage = stages.get(task_id, {})
        model_ids = [str(item) for item in stage.get("model_ids", []) if str(item)]
        matched_tables = _matching_tables(deliverable, model_ids, table_entries)
        paper_sections = _matching_paper_sections(deliverable, index, paper_text)

        model_ok = bool(model_ids) or task_type == "exploration"
        table_ok = bool(matched_tables)
        paper_ok = paper_text is None or bool(paper_sections)
        status = "complete" if model_ok and table_ok and paper_ok else "incomplete"

        if not model_ok:
            issues.append(f"{task_id}: missing executable model binding")
        if not table_ok:
            issues.append(f"{task_id}: missing result table binding")
        if paper_text is not None and not paper_ok:
            issues.append(f"{task_id}: missing paper section binding")

        items.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "objective": deliverable.objective,
                "model_ids": model_ids,
                "tables": matched_tables,
                "paper_sections": paper_sections,
                "model_ok": model_ok,
                "table_ok": table_ok,
                "paper_ok": paper_ok,
                "status": status,
            }
        )

    complete_count = sum(1 for item in items if item["status"] == "complete")
    coverage_pct = round(100.0 * complete_count / len(items), 1) if items else 100.0
    return {
        "schema_version": "1.0",
        "task_count": len(items),
        "complete_count": complete_count,
        "coverage_pct": coverage_pct,
        "paper_checked": paper_text is not None,
        "passed": not issues,
        "issues": issues,
        "items": items,
    }


def write_task_traceability_report(workspace, report: dict[str, Any]) -> Path:
    return write_text(
        workspace.logs_dir / "task_traceability_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def task_traceability_blocking_issues(report: dict[str, Any]) -> list[str]:
    if report.get("passed") is True:
        return []
    return [str(item) for item in report.get("issues", []) if str(item)]


def _stages_by_id(formulation: FormulationSpec | None) -> dict[str, dict[str, Any]]:
    if formulation is None:
        return {}
    stages: dict[str, dict[str, Any]] = {}
    for stage in formulation.stages:
        stage_id = str(stage.get("stage_id") or "")
        if stage_id:
            stages[stage_id] = stage
    return stages


def _table_entries(registry: ResultRegistry | None) -> list[dict[str, Any]]:
    if registry is None:
        return []
    return [entry for entry in registry.entries if entry.get("type") == "table"]


def _matching_tables(
    deliverable: TaskDeliverableSpec,
    model_ids: list[str],
    table_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    task_type = deliverable.task_type or "exploration"
    required_tokens = [
        str(token).lower()
        for token in deliverable.required_tables
        if str(token).strip()
    ]
    task_tokens = list(_TASK_TYPE_TABLE_TOKENS.get(task_type, ()))
    model_tokens = [model_id.lower() for model_id in model_ids]
    tokens = [*model_tokens, *required_tokens, *task_tokens]

    matches: list[dict[str, str]] = []
    for entry in table_entries:
        name = str(entry.get("name") or "").lower()
        model_id = str(entry.get("model_id") or "").lower()
        path = str(entry.get("path") or "")
        if _matches_any(name, tokens) or (model_id and model_id in model_tokens):
            matches.append({"name": str(entry.get("name") or ""), "path": path})
    return _dedupe_table_matches(matches)


def _matching_paper_sections(
    deliverable: TaskDeliverableSpec,
    index: int,
    paper_text: str | None,
) -> list[str]:
    if not paper_text:
        return []
    task_id = deliverable.task_id or f"Q{index}"
    task_number = _task_number(task_id) or index
    objective_tokens = _objective_tokens(deliverable.objective)
    markers = [
        task_id,
        task_id.lower(),
        f"Q{task_number}",
        f"q{task_number}",
        f"问题{task_number}",
        f"问题 {task_number}",
        f"问题{_cn_number(task_number)}",
        f"第{_cn_number(task_number)}问",
    ]
    sections = _split_sections(paper_text)
    matches: list[str] = []
    for heading, body in sections:
        haystack = f"{heading}\n{body}".lower()
        marker_hit = any(marker.lower() in haystack for marker in markers if marker)
        objective_hit = bool(objective_tokens) and sum(
            1 for token in objective_tokens if token.lower() in haystack
        ) >= min(2, len(objective_tokens))
        if marker_hit or objective_hit:
            matches.append(heading or "body")
    return list(dict.fromkeys(matches))


def _matches_any(text: str, tokens: list[str]) -> bool:
    return any(token and token in text for token in tokens)


def _objective_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text or "")
    stop = {"根据", "附件", "数据", "建立", "模型", "进行", "分析", "给出", "完成"}
    return [token for token in tokens[:8] if token not in stop]


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            if current_heading or current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_heading or current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
    return sections or [("body", text)]


def _cn_number(index: int) -> str:
    mapping = {
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    return mapping.get(index, str(index))


def _task_number(task_id: str) -> int | None:
    match = re.search(r"(\d+)", task_id or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _dedupe_table_matches(matches: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for match in matches:
        key = match.get("path") or match.get("name") or ""
        if key and key not in seen:
            seen.add(key)
            deduped.append(match)
    return deduped
