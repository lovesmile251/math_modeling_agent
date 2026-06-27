from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.base import K_EXECUTION_STATUS, K_PAPER_QUALITY_SCORE, K_PREWRITING_GATE_STATUS, WorkflowPhase, WorkflowState
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "rerun_from_phase": self.rerun_from_phase.value,
            "invalidated_phases": [phase.value for phase in self.invalidated_phases],
            "actions": list(self.actions),
            "refresh_artifacts": list(self.refresh_artifacts),
            "can_auto_apply": self.can_auto_apply,
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

    if state.notes.get("traceability_gate") == "failed":
        return ReworkRoute(
            WorkflowPhase.SECTION_WRITING,
            "paper contains numeric claims that are not traceable to result evidence",
            severity="high",
        )

    if state.notes.get("export_pdf_layout_gate") == "failed":
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
    )


def write_rework_plan(workspace: Any, plan: ReworkPlan) -> Path:
    workspace_logs_dir = getattr(workspace, "logs_dir", None)
    logs_dir = Path(workspace_logs_dir) if workspace_logs_dir is not None else Path(workspace) / "logs"
    return write_text(
        logs_dir / "auto_rework_plan.json",
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
    )


def _paper_score(state: WorkflowState) -> int | None:
    raw = state.notes.get(K_PAPER_QUALITY_SCORE)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


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
        WorkflowPhase.MODEL_DECISION: ("model_decision", "formulation_spec"),
        WorkflowPhase.EXPERIMENT_PLAN: ("experiment_plan",),
        WorkflowPhase.CODE_PLAN: ("code_plan",),
        WorkflowPhase.CODE_GENERATION: ("code",),
        WorkflowPhase.EXECUTION: ("execution_log", "result_registry"),
        WorkflowPhase.RESULT_ANALYSIS: ("result_analysis", "result_registry"),
        WorkflowPhase.EVIDENCE_MAPPING: ("claim_evidence_map",),
        WorkflowPhase.PAPER_OUTLINE: ("paper_outline",),
        WorkflowPhase.SECTION_WRITING: ("section_draft", "paper"),
        WorkflowPhase.FACT_REVIEW: ("review_findings",),
        WorkflowPhase.MATH_REVIEW: ("review_findings",),
        WorkflowPhase.STRUCTURE_REVIEW: ("review_findings",),
        WorkflowPhase.LANGUAGE_REVIEW: ("paper_quality", "review"),
        WorkflowPhase.EXPORT: ("paper_pdf_layout_report", "paper"),
    }
    artifacts: list[str] = []
    for phase in phases:
        artifacts.extend(phase_artifacts.get(phase, ()))
    return tuple(dict.fromkeys(artifacts))
