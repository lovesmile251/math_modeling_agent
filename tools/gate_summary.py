from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.base import (
    K_AUTO_REWORK_STATUS,
    K_EXPORT_BLOCKING_ISSUES,
    K_EXPORT_ERRORS,
    K_PAPER_QUALITY_SCORE,
    QUALITY_GATE_NOTE_KEYS,
    WorkflowState,
)
from tools.file_tool import write_text
from tools.rework_router import build_rework_plan


def build_workflow_gate_summary(state: WorkflowState) -> dict[str, Any]:
    """Summarize the final gate state of one workflow run."""

    rework_plan = build_rework_plan(state)
    gates = {
        key: state.notes[key]
        for key in QUALITY_GATE_NOTE_KEYS
        if state.notes.get(key)
    }
    blockers = _blockers(state.notes)
    return {
        "schema_version": "1.0",
        "run_id": state.run_id,
        "workspace": str(state.workspace.root),
        "phase": state.phase.value,
        "paper_quality_score": _number_or_text(state.notes.get(K_PAPER_QUALITY_SCORE)),
        "gates": gates,
        "passed_gates": sorted(key for key, value in gates.items() if str(value) == "passed"),
        "failed_gates": sorted(
            key
            for key, value in gates.items()
            if str(value).lower() in {"failed", "blocked", "fail"}
        ),
        "blockers": blockers,
        "auto_rework": {
            "status": state.notes.get(K_AUTO_REWORK_STATUS, ""),
            "attempts_used": state.notes.get("auto_rework_attempts_used", "0"),
            "last_reason": state.notes.get("auto_rework_last_reason", ""),
            "remaining_reason": state.notes.get("auto_rework_remaining_reason", ""),
        },
        "recommended_rework": rework_plan.to_dict() if rework_plan else None,
        "artifacts": {
            key: str(path)
            for key, path in sorted(state.artifacts.items())
            if Path(path).exists()
        },
    }


def format_workflow_gate_summary(summary: dict[str, Any]) -> str:
    gates = summary.get("gates") if isinstance(summary.get("gates"), dict) else {}
    blockers = summary.get("blockers") if isinstance(summary.get("blockers"), dict) else {}
    auto = summary.get("auto_rework") if isinstance(summary.get("auto_rework"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    recommended = (
        summary.get("recommended_rework")
        if isinstance(summary.get("recommended_rework"), dict)
        else None
    )

    lines = [
        "# Workflow Gate Summary",
        "",
        f"- Workspace: `{summary.get('workspace', '')}`",
        f"- Final phase: `{summary.get('phase', '')}`",
        f"- Paper quality score: {summary.get('paper_quality_score', '')}",
        f"- Failed gates: {len(summary.get('failed_gates', []) or [])}",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    if gates:
        for key, value in sorted(gates.items()):
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("| _none recorded_ |  |")

    lines.extend(["", "## Blockers"])
    if blockers:
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(blockers.items()))
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Auto Rework",
            f"- Status: {auto.get('status', '') or 'not_run'}",
            f"- Attempts used: {auto.get('attempts_used', '0')}",
        ]
    )
    if auto.get("last_reason"):
        lines.append(f"- Last reason: {auto['last_reason']}")
    if auto.get("remaining_reason"):
        lines.append(f"- Remaining reason: {auto['remaining_reason']}")

    lines.extend(["", "## Recommended Rework"])
    if recommended:
        route = recommended.get("route") if isinstance(recommended.get("route"), dict) else {}
        lines.append(f"- Rerun from: `{recommended.get('rerun_from_phase', '')}`")
        lines.append(f"- Severity: {route.get('severity', '')}")
        lines.append(f"- Blocking: {route.get('blocking', '')}")
        lines.append(f"- Reason: {route.get('reason', '')}")
    else:
        lines.append("- None")

    lines.extend(["", "## Key Artifacts"])
    for key in (
        "paper",
        "paper_quality",
        "task_traceability_report",
        "innovation_evidence_report",
        "auto_rework_report",
        "workflow_gate_summary",
    ):
        if key in artifacts:
            lines.append(f"- `{key}`: `{artifacts[key]}`")
    return "\n".join(lines)


def write_workflow_gate_summary(state: WorkflowState) -> dict[str, Path]:
    summary = build_workflow_gate_summary(state)
    json_path = write_text(
        state.workspace.logs_dir / "workflow_gate_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    md_path = write_text(
        state.workspace.logs_dir / "workflow_gate_summary.md",
        format_workflow_gate_summary(summary),
    )
    return {"json": json_path, "markdown": md_path}


def _blockers(notes: dict[str, str]) -> dict[str, str]:
    suffixes = ("_blocking_issues", "_issues", "_warnings")
    out = {
        key: str(value)
        for key, value in notes.items()
        if value and any(key.endswith(suffix) for suffix in suffixes)
    }
    if notes.get(K_EXPORT_ERRORS):
        out[K_EXPORT_ERRORS] = notes[K_EXPORT_ERRORS]
    if notes.get(K_EXPORT_BLOCKING_ISSUES):
        out[K_EXPORT_BLOCKING_ISSUES] = notes[K_EXPORT_BLOCKING_ISSUES]
    return out


def _number_or_text(value: str | None) -> int | float | str:
    if value is None:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number
