from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.base import (
    A_CLAIM_EVIDENCE_MAP,
    A_CODE,
    A_CODE_PLAN,
    A_EXPERIMENT_PLAN,
    A_EXPERIMENT_REPORT,
    A_EXECUTION_LOG,
    A_FORMULATION_SPEC,
    A_INNOVATION_EVIDENCE_REPORT,
    A_MODEL_DECISION,
    A_PAPER,
    A_PAPER_OUTLINE,
    A_PAPER_PDF_LAYOUT_REPORT,
    A_PAPER_QUALITY,
    A_RESULT_REGISTRY,
    A_REVIEW,
    A_REVIEW_FINDINGS,
    A_SECTION_DRAFT,
    A_TASK_TRACEABILITY_REPORT,
    K_AUTO_REWORK_REPAIR_BRIEF,
    K_AUTO_REWORK_REPAIR_HINTS,
    K_AUTO_REWORK_RERUN_FROM_PHASE,
    K_EXPORT_PDF_LAYOUT_GATE,
    K_EXPORT_QUALITY_GATE,
    K_EXECUTION_STATUS,
    K_INNOVATION_EVIDENCE_GATE,
    K_PAPER_EVIDENCE_GATE,
    K_PAPER_EVIDENCE_ISSUES,
    K_PAPER_QUALITY_SCORE,
    K_PREWRITING_GATE_STATUS,
    K_STRONG_BASELINE_GATE,
    K_STRONG_BASELINE_ISSUES,
    K_TASK_TRACEABILITY_BLOCKING_ISSUES,
    K_TASK_TRACEABILITY_GATE,
    QUALITY_GATE_NOTE_KEYS,
    PhaseStatus,
    WorkflowPhase,
    WorkflowState,
)
from tools.file_tool import write_text


@dataclass(frozen=True)
class ReworkRoute:
    target_phase: WorkflowPhase
    reason: str
    severity: str = "medium"
    blocking: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        payload = asdict(self)
        payload["target_phase"] = self.target_phase.value
        return payload


@dataclass(frozen=True)
class ReworkPlan:
    route: ReworkRoute
    rerun_from_phase: WorkflowPhase
    invalidated_phases: tuple[WorkflowPhase, ...]
    actions: tuple[str, ...]
    refresh_artifacts: tuple[str, ...]
    can_auto_apply: bool = False
    repair_hints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "rerun_from_phase": self.rerun_from_phase.value,
            "invalidated_phases": [phase.value for phase in self.invalidated_phases],
            "actions": list(self.actions),
            "refresh_artifacts": list(self.refresh_artifacts),
            "can_auto_apply": self.can_auto_apply,
            "repair_hints": list(self.repair_hints),
        }


@dataclass(frozen=True)
class ReworkApplyResult:
    applied: bool
    rerun_from_phase: WorkflowPhase | None
    invalidated_phases: tuple[WorkflowPhase, ...]
    stale_artifacts: tuple[str, ...]
    removed_artifacts: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "rerun_from_phase": self.rerun_from_phase.value if self.rerun_from_phase else None,
            "invalidated_phases": [phase.value for phase in self.invalidated_phases],
            "stale_artifacts": list(self.stale_artifacts),
            "removed_artifacts": list(self.removed_artifacts),
            "reason": self.reason,
        }


def recommend_rework_route(state: WorkflowState) -> ReworkRoute | None:
    """Return the earliest useful workflow phase to rerun for current failures."""

    if state.notes.get(K_EXECUTION_STATUS) == "failed":
        return ReworkRoute(
            WorkflowPhase.CODE_PLAN,
            "code execution failed; revise code plan and regenerate executable analysis",
            severity="high",
        )

    formulation_status = state.notes.get("formulation_status")
    if formulation_status == "needs_revision":
        return ReworkRoute(
            WorkflowPhase.MODEL_DECISION,
            "formulation spec has validation issues; revisit selected models and task-model compatibility",
            severity="high",
        )

    if state.notes.get(K_PREWRITING_GATE_STATUS) == "blocked":
        report = state.notes.get("prewriting_gate_report", "")
        if "优化结果表" in report or "optimization_result" in report:
            return ReworkRoute(
                WorkflowPhase.MODEL_DECISION,
                "pre-writing gate requires missing optimization deliverables",
                severity="high",
            )
        return ReworkRoute(
            WorkflowPhase.RESULT_ANALYSIS,
            "pre-writing gate lacks enough result evidence for paper writing",
            severity="high",
        )

    if state.notes.get(K_STRONG_BASELINE_GATE) == "failed":
        issues = state.notes.get(K_STRONG_BASELINE_ISSUES, "")
        if "no baseline" in issues.lower():
            return ReworkRoute(
                WorkflowPhase.MODEL_DECISION,
                "strong baseline gate failed because no baseline model was designated",
                severity="high",
            )
        return ReworkRoute(
            WorkflowPhase.EXPERIMENT_PLAN,
            "strong baseline gate failed; add executed baseline comparison, validation, and ablation evidence",
            severity="high",
        )

    if state.notes.get(K_INNOVATION_EVIDENCE_GATE) == "failed":
        return ReworkRoute(
            WorkflowPhase.EXPERIMENT_PLAN,
            "innovation evidence gate failed; run or remove unsupported innovation claims",
            severity="high",
        )

    if state.notes.get(K_PAPER_EVIDENCE_GATE) == "failed":
        route = _route_for_paper_evidence_issues(state.notes.get(K_PAPER_EVIDENCE_ISSUES, ""), state)
        if route is not None:
            return route
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "paper evidence audit failed; revise paper sections with table-backed quantitative evidence",
            severity="high",
        )

    if state.notes.get(K_TASK_TRACEABILITY_GATE) == "failed":
        issues = state.notes.get(K_TASK_TRACEABILITY_BLOCKING_ISSUES, "")
        lowered = issues.lower()
        if "missing executable model binding" in lowered:
            return ReworkRoute(
                WorkflowPhase.MODEL_DECISION,
                "task traceability gate lacks model bindings for one or more deliverables",
                severity="high",
            )
        if "missing result table binding" in lowered:
            return ReworkRoute(
                WorkflowPhase.EVIDENCE_MAPPING,
                "task traceability gate lacks result-table bindings for one or more deliverables",
                severity="high",
            )
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "task traceability gate lacks paper-section bindings for one or more deliverables",
            severity="high",
        )

    if state.notes.get("traceability_gate") == "failed":
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "paper contains numeric claims that are not traceable to result evidence",
            severity="high",
        )

    if state.notes.get(K_EXPORT_QUALITY_GATE) == "failed":
        specific_route = _route_for_export_quality_blockers(state)
        if specific_route is not None:
            return specific_route
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "formal export quality gate failed; revise paper sections and remove blocking issues",
            severity="high",
        )

    if state.notes.get(K_EXPORT_PDF_LAYOUT_GATE) == "failed":
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "PDF render layout check failed; revise paper tables, code blocks, or embedded assets",
            severity="medium",
        )

    score = _paper_score(state)
    if score is not None and score < 82:
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            f"paper quality score {score} is below the 82-point gate",
            severity="medium",
            blocking=False,
        )

    return None


def build_rework_plan(state: WorkflowState) -> ReworkPlan | None:
    """Build a serializable rerun plan from the recommended route."""

    route = recommend_rework_route(state)
    if route is None:
        return None
    invalidated = _invalidated_phases(route.target_phase)
    return ReworkPlan(
        route=route,
        rerun_from_phase=route.target_phase,
        invalidated_phases=invalidated,
        actions=_actions_for_phase(route.target_phase),
        refresh_artifacts=_artifacts_for_phases(invalidated),
        can_auto_apply=route.blocking and route.severity in {"high", "medium"},
        repair_hints=_repair_hints_for_state(state, route),
    )


def write_rework_plan(workspace: Any, plan: ReworkPlan) -> Path:
    workspace_logs_dir = getattr(workspace, "logs_dir", None)
    logs_dir = Path(workspace_logs_dir) if workspace_logs_dir is not None else Path(workspace) / "logs"
    return write_text(
        logs_dir / "auto_rework_plan.json",
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
    )


def build_auto_rework_report(
    *,
    plan: ReworkPlan,
    apply_result: ReworkApplyResult,
    final_plan: ReworkPlan | None,
    before_notes: dict[str, str],
    after_notes: dict[str, str],
    attempt: int,
    max_attempts: int,
    status: str,
) -> dict[str, Any]:
    """Build a readable execution report for an automatic rework attempt."""

    return {
        "schema_version": "1.0",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "status": status,
        "initial_route": plan.route.to_dict(),
        "applied": apply_result.to_dict(),
        "actions": list(plan.actions),
        "repair_hints": list(plan.repair_hints),
        "refresh_artifacts": list(plan.refresh_artifacts),
        "before_gates": _gate_snapshot(before_notes),
        "after_gates": _gate_snapshot(after_notes),
        "before_blockers": _blocker_snapshot(before_notes),
        "after_blockers": _blocker_snapshot(after_notes),
        "remaining_route": final_plan.route.to_dict() if final_plan else None,
        "remaining_plan": final_plan.to_dict() if final_plan else None,
    }


def format_auto_rework_report(report: dict[str, Any]) -> str:
    route = report.get("initial_route") if isinstance(report.get("initial_route"), dict) else {}
    remaining = report.get("remaining_route") if isinstance(report.get("remaining_route"), dict) else None
    lines = [
        "# 自动返工报告",
        "",
        f"- 状态：{report.get('status', 'unknown')}",
        f"- 尝试次数：{report.get('attempt', 0)}/{report.get('max_attempts', 0)}",
        f"- 返工起点：{route.get('target_phase', '')}",
        f"- 严重程度：{route.get('severity', '')}",
        f"- 触发原因：{route.get('reason', '')}",
        "",
        "## 执行动作",
    ]
    actions = report.get("actions") if isinstance(report.get("actions"), list) else []
    lines.extend(f"- {item}" for item in actions)
    repair_hints = report.get("repair_hints") if isinstance(report.get("repair_hints"), list) else []
    lines.extend(["", "## Repair Hints"])
    lines.extend(f"- {item}" for item in repair_hints) if repair_hints else lines.append("- None")
    refresh = report.get("refresh_artifacts") if isinstance(report.get("refresh_artifacts"), list) else []
    lines.extend(["", "## 刷新产物"])
    lines.extend(f"- `{item}`" for item in refresh) if refresh else lines.append("- 无")

    lines.extend(["", "## 门禁变化", "", "| 门禁 | 返工前 | 返工后 |", "|---|---|---|"])
    before_gates = report.get("before_gates") if isinstance(report.get("before_gates"), dict) else {}
    after_gates = report.get("after_gates") if isinstance(report.get("after_gates"), dict) else {}
    for key in sorted(set(before_gates) | set(after_gates)):
        lines.append(f"| `{key}` | {before_gates.get(key, '')} | {after_gates.get(key, '')} |")

    lines.extend(["", "## 阻塞项变化"])
    before_blockers = report.get("before_blockers") if isinstance(report.get("before_blockers"), dict) else {}
    after_blockers = report.get("after_blockers") if isinstance(report.get("after_blockers"), dict) else {}
    if before_blockers:
        lines.append("")
        lines.append("返工前：")
        lines.extend(f"- `{key}`: {value}" for key, value in before_blockers.items())
    if after_blockers:
        lines.append("")
        lines.append("返工后：")
        lines.extend(f"- `{key}`: {value}" for key, value in after_blockers.items())
    if not before_blockers and not after_blockers:
        lines.append("- 无")

    lines.extend(["", "## 剩余状态"])
    if remaining:
        lines.append(f"- 仍需返工：{remaining.get('target_phase', '')}")
        lines.append(f"- 剩余原因：{remaining.get('reason', '')}")
    else:
        lines.append("- 已解决当前自动返工阻塞项")
    return "\n".join(lines)


def write_auto_rework_report(workspace: Any, report: dict[str, Any]) -> dict[str, Path]:
    workspace_logs_dir = getattr(workspace, "logs_dir", None)
    logs_dir = Path(workspace_logs_dir) if workspace_logs_dir is not None else Path(workspace) / "logs"
    json_path = write_text(
        logs_dir / "auto_rework_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    md_path = write_text(logs_dir / "auto_rework_report.md", format_auto_rework_report(report))
    return {"json": json_path, "markdown": md_path}


def apply_rework_plan(
    state: WorkflowState,
    plan: ReworkPlan | None = None,
    *,
    clear_artifacts: bool = False,
    operator: str = "system",
) -> ReworkApplyResult:
    """Apply a rework plan to workflow state without running agents.

    This prepares the state for a rerun by marking downstream phases as
    ``needs_revision`` and recording a decision. Artifact files are not deleted;
    optionally their state mappings can be removed so downstream agents refresh
    them.
    """

    resolved = plan or build_rework_plan(state)
    if resolved is None:
        return ReworkApplyResult(False, None, (), (), (), "no rework plan available")

    for phase in resolved.invalidated_phases:
        state.set_phase_status(phase, PhaseStatus.NEEDS_REVISION)
    state.phase = resolved.rerun_from_phase
    removed: list[str] = []
    stale = tuple(name for name in resolved.refresh_artifacts if name in state.artifacts)
    if clear_artifacts:
        for artifact_name in resolved.refresh_artifacts:
            if artifact_name in state.artifacts:
                state.artifacts.pop(artifact_name, None)
                removed.append(artifact_name)

    state.notes["auto_rework_applied"] = "true"
    state.notes[K_AUTO_REWORK_RERUN_FROM_PHASE] = resolved.rerun_from_phase.value
    state.notes["auto_rework_invalidated_phases"] = ",".join(
        phase.value for phase in resolved.invalidated_phases
    )
    state.notes["auto_rework_stale_artifacts"] = ",".join(stale)
    if resolved.repair_hints:
        state.notes[K_AUTO_REWORK_REPAIR_HINTS] = json.dumps(
            list(resolved.repair_hints),
            ensure_ascii=False,
        )
        state.notes[K_AUTO_REWORK_REPAIR_BRIEF] = "; ".join(resolved.repair_hints[:3])
    state.record_decision(
        resolved.rerun_from_phase,
        "apply_rework_plan",
        operator=operator,
        notes=resolved.route.reason,
    )
    return ReworkApplyResult(
        True,
        resolved.rerun_from_phase,
        resolved.invalidated_phases,
        stale,
        tuple(removed),
        resolved.route.reason,
    )


def _gate_snapshot(notes: dict[str, str]) -> dict[str, str]:
    return {
        key: str(notes[key])
        for key in (*QUALITY_GATE_NOTE_KEYS, "traceability_gate")
        if notes.get(key)
    }


def _blocker_snapshot(notes: dict[str, str]) -> dict[str, str]:
    blocker_suffixes = (
        "_issues",
        "_blocking_issues",
        "_errors",
        "_stop_reason",
    )
    return {
        key: str(value)
        for key, value in notes.items()
        if value and any(key.endswith(suffix) for suffix in blocker_suffixes)
    }


def _paper_score(state: WorkflowState) -> int | None:
    raw = state.notes.get(K_PAPER_QUALITY_SCORE)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _route_for_export_quality_blockers(state: WorkflowState) -> ReworkRoute | None:
    blockers = " ".join(
        str(state.notes.get(key, ""))
        for key in (
            "export_blocking_issues",
            "paper_quality_report",
            "export_errors",
        )
    )
    lowered = blockers.lower()
    if not lowered.strip():
        return None

    if (
        "claimed high-level model has no matching generated result table" in lowered
        or "selected high-level model has no matching generated result table" in lowered
        or "high-level model table lacks model-specific metrics" in lowered
        or "model claims without generated result tables" in lowered
    ):
        return ReworkRoute(
            WorkflowPhase.CODE_PLAN,
            "high-level model evidence table is missing or lacks required metrics; rerun code planning and execution to produce table-backed model diagnostics",
            severity="high",
        )

    if "core result table missing" in lowered:
        has_tables = bool(list(state.workspace.tables_dir.glob("*.csv")))
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING if has_tables else WorkflowPhase.RESULT_ANALYSIS,
            (
                "paper result section lacks a core table; cite generated tables in the paper"
                if has_tables
                else "paper result section lacks a core table and no generated tables are available; rebuild result analysis"
            ),
            severity="high",
        )

    if "risk model evidence weak" in lowered or "selected high-level model missing from paper narrative" in lowered:
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "high-level risk/mechanism model is claimed without model-specific metrics; rewrite model and result sections with table-backed metrics",
            severity="high",
        )

    if "award evidence density weak" in lowered:
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "abstract lacks enough quantitative evidence; rewrite abstract and conclusion using generated result values",
            severity="medium",
            blocking=True,
        )

    if (
        "award structure weak" in lowered
        or "problem-answer closure weak" in lowered
        or "model formulation weak" in lowered
        or "model validation section weak" in lowered
        or "conclusion answer density weak" in lowered
    ):
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "national-award paper structure is incomplete; rewrite high-value sections and close every subproblem answer thread",
            severity="high",
        )

    return None


def _route_for_paper_evidence_issues(issues: str, state: WorkflowState) -> ReworkRoute | None:
    lowered = str(issues).lower()
    if not lowered.strip():
        return None
    if (
        "claimed high-level model has no matching generated result table" in lowered
        or "selected high-level model has no matching generated result table" in lowered
        or "high-level model table lacks model-specific metrics" in lowered
    ):
        return ReworkRoute(
            WorkflowPhase.CODE_PLAN,
            "paper evidence audit found high-level model table evidence missing or incomplete",
            severity="high",
        )
    if "core result table missing" in lowered:
        has_tables = bool(list(state.workspace.tables_dir.glob("*.csv")))
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING if has_tables else WorkflowPhase.RESULT_ANALYSIS,
            (
                "paper evidence audit found uncited generated tables"
                if has_tables
                else "paper evidence audit found no result table evidence to cite"
            ),
            severity="high",
        )
    if (
        "risk model evidence weak" in lowered
        or "award evidence density weak" in lowered
        or "selected high-level model missing from paper narrative" in lowered
    ):
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "paper evidence audit requires stronger model-specific metrics in paper sections",
            severity="high",
        )
    return None


def _repair_hints_for_state(state: WorkflowState, route: ReworkRoute) -> tuple[str, ...]:
    hints: list[str] = []
    notes_blob = " ".join(str(value) for value in state.notes.values()).lower()

    if route.target_phase == WorkflowPhase.CODE_PLAN:
        hints.extend(
            [
                "Produce non-empty CSV result tables for every selected executable model.",
                "Update generated code so model-specific diagnostics are written as table columns, not only prose.",
            ]
        )
        if "cvar" in notes_blob:
            hints.append("For cvar_optimization, output var_loss, cvar_loss, tail_scenario_count, and risk_adjusted_score.")
        if "robust" in notes_blob:
            hints.append("For robust_optimization, output robust_value, robust_resource, capacity_slack, and uncertainty_rate.")
        if "chance" in notes_blob or "service_level" in notes_blob:
            hints.append("For chance_constrained_optimization, output safe_resource, service_level, feasibility_probability, and violation_probability.")
        if "execution" in route.reason.lower() or state.notes.get(K_EXECUTION_STATUS) == "failed":
            error = str(state.notes.get("execution_error", "")).strip()
            if error:
                hints.append("Resolve the recorded execution error before regenerating downstream evidence: " + error[:240])

    elif route.target_phase == WorkflowPhase.EXPERIMENT_PLAN:
        hints.extend(
            [
                "Add an executed baseline comparison and require strong_baseline_audit.passed in experiment_report.json.",
                "Add validation, ablation, and sensitivity evidence for any claimed innovation or high-level model.",
            ]
        )
        if state.notes.get(K_INNOVATION_EVIDENCE_GATE) == "failed":
            hints.append("Either execute the claimed innovation model or remove unsupported innovation claims from the paper.")

    elif route.target_phase == WorkflowPhase.RESULT_ANALYSIS:
        hints.extend(
            [
                "Recompute result summaries from generated tables and expose final task answers as numeric rows.",
                "Ensure result_analysis names the source table for each core conclusion.",
            ]
        )

    elif route.target_phase == WorkflowPhase.EVIDENCE_MAPPING:
        hints.extend(
            [
                "Rebuild result_registry, claim_evidence_map, and task_traceability_report from the latest tables.",
                "For each deliverable, bind an executable model, a result table, and a paper section.",
            ]
        )
        task_issues = str(state.notes.get(K_TASK_TRACEABILITY_BLOCKING_ISSUES, "")).strip()
        if task_issues:
            hints.append("Resolve task traceability blockers: " + task_issues[:240])

    elif route.target_phase == WorkflowPhase.SECTION_WRITING:
        hints.extend(
            [
                "Rewrite affected paper sections around generated table values and avoid unsupported model claims.",
                "Add at least one Markdown core result table in the Results section and interpret it in prose.",
                "Put at least five substantive numeric result values in the abstract when available.",
            ]
        )
        if "selected high-level model missing from paper narrative" in notes_blob:
            hints.append("Add a model/results paragraph for each selected high-level model and cite its table-backed metrics.")
        if "risk model evidence weak" in notes_blob:
            hints.append("For each risk model claim, mention at least two model-specific metrics from its generated table.")
        if "award structure weak" in notes_blob or "problem-answer closure weak" in notes_blob:
            hints.append("Restore the national-contest section skeleton and close each Q1/Q2/Q3 thread with method, table-backed result, and conclusion.")
        if "model validation section weak" in notes_blob:
            hints.append("Add a validation subsection with error analysis, baseline comparison, ablation, or robustness evidence.")
        if "conclusion answer density weak" in notes_blob:
            hints.append("Rewrite the conclusion as numbered final answers with quantitative values for each subproblem.")

    elif route.target_phase == WorkflowPhase.EXPORT:
        hints.extend(
            [
                "Run export after paper regeneration so export_quality_gate, paper_evidence_gate, and PDF layout gate are evaluated.",
                "Ensure paper_evidence_audit.json and paper_evidence_audit.md are written before accepting final delivery.",
            ]
        )

    hints.append("After rerun, compare the same gate status and blocker text to avoid repeated same-cause loops.")
    return tuple(_dedupe(hints))


def _invalidated_phases(target_phase: WorkflowPhase) -> tuple[WorkflowPhase, ...]:
    terminal = WorkflowPhase.COMPLETE.order
    return tuple(
        phase
        for phase in WorkflowPhase
        if target_phase.order <= phase.order < terminal
    )


def _actions_for_phase(target_phase: WorkflowPhase) -> tuple[str, ...]:
    actions_by_phase: dict[WorkflowPhase, tuple[str, ...]] = {
        WorkflowPhase.MODEL_DECISION: (
            "revise model selection and task-model compatibility",
            "refresh formulation, experiment plan, and code plan",
            "rerun executable model analysis",
            "rebuild result evidence and regenerate paper sections",
        ),
        WorkflowPhase.EXPERIMENT_PLAN: (
            "revise validation, baseline, ablation, and innovation-evidence requirements",
            "rerun executable model analysis with required validation artifacts",
            "rebuild experiment report and downstream evidence",
            "regenerate affected paper sections",
        ),
        WorkflowPhase.CODE_PLAN: (
            "regenerate code plan from current formulation",
            "rerun code generation and execution",
            "refresh result registry and downstream evidence",
        ),
        WorkflowPhase.RESULT_ANALYSIS: (
            "rerun result analysis against generated tables and figures",
            "rebuild evidence registry and claim map",
            "regenerate affected paper sections",
        ),
        WorkflowPhase.EVIDENCE_MAPPING: (
            "rebuild result registry and task traceability mappings",
            "refresh claim evidence map",
            "regenerate paper sections that cite affected evidence",
        ),
        WorkflowPhase.SECTION_WRITING: (
            "regenerate paper outline or affected sections",
            "rerun fact, math, structure, and language review",
            "export the final document and rerun PDF layout checks",
        ),
    }
    return actions_by_phase.get(
        target_phase,
        (
            f"rerun workflow from {target_phase.value}",
            "refresh downstream artifacts",
        ),
    )


def _artifacts_for_phases(phases: tuple[WorkflowPhase, ...]) -> tuple[str, ...]:
    phase_artifacts: dict[WorkflowPhase, tuple[str, ...]] = {
        WorkflowPhase.MODEL_DECISION: (A_MODEL_DECISION, A_FORMULATION_SPEC),
        WorkflowPhase.EXPERIMENT_PLAN: (A_EXPERIMENT_PLAN, A_EXPERIMENT_REPORT),
        WorkflowPhase.CODE_PLAN: (A_CODE_PLAN,),
        WorkflowPhase.CODE_GENERATION: (A_CODE,),
        WorkflowPhase.EXECUTION: (A_EXECUTION_LOG, A_RESULT_REGISTRY),
        WorkflowPhase.RESULT_ANALYSIS: ("result_analysis", A_RESULT_REGISTRY),
        WorkflowPhase.EVIDENCE_MAPPING: (
            A_CLAIM_EVIDENCE_MAP,
            A_TASK_TRACEABILITY_REPORT,
            A_INNOVATION_EVIDENCE_REPORT,
        ),
        WorkflowPhase.PAPER_OUTLINE: (A_PAPER_OUTLINE,),
        WorkflowPhase.SECTION_WRITING: (A_SECTION_DRAFT, A_PAPER),
        WorkflowPhase.FACT_REVIEW: (A_REVIEW_FINDINGS,),
        WorkflowPhase.MATH_REVIEW: (A_REVIEW_FINDINGS,),
        WorkflowPhase.STRUCTURE_REVIEW: (A_REVIEW_FINDINGS,),
        WorkflowPhase.LANGUAGE_REVIEW: (A_PAPER_QUALITY, A_REVIEW),
        WorkflowPhase.EXPORT: (A_PAPER_PDF_LAYOUT_REPORT, A_PAPER),
    }
    artifacts: list[str] = []
    for phase in phases:
        artifacts.extend(phase_artifacts.get(phase, ()))
    return tuple(dict.fromkeys(artifacts))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
