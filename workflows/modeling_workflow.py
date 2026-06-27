from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from agents.analysis_agent import AnalysisAgent
from agents.base import (
    A_CODE,
    K_EXECUTION_STATUS,
    K_LLM_STATUS,
    K_LLM_FAILURE_KIND,
    K_PAPER_QUALITY_SCORE,
    K_PREWRITING_GATE_STATUS,
    K_REVIEW_REPORT,
    PhaseStatus,
    WorkflowPhase,
    WorkflowState,
)
from agents.code_plan_agent import CodePlanAgent
from agents.coding_agent import CodingAgent
from agents.decision_agent import DecisionAgent
from agents.evidence_agent import EvidenceAgent
from agents.execution_agent import ExecutionAgent
from agents.experiment_plan_agent import ExperimentPlanAgent
from agents.formulation_agent import FormulationAgent
from agents.export_agent import ExportAgent
from agents.fact_reviewer import FactReviewerAgent
from agents.language_reviewer import LanguageReviewerAgent
from agents.math_reviewer import MathReviewerAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.modeling_agent import ModelingAgent
from agents.modeling_critic_agent import ModelingCriticAgent
from agents.outline_agent import OutlineAgent
from agents.problem_agent import ProblemAgent
from agents.review_agent import ReviewAgent
from agents.structure_reviewer import StructureReviewerAgent
from agents.writing_agent import WritingAgent
from app.config import WORKSPACE
from app.config import PROJECT_ROOT, WorkspaceConfig
from tools.file_tool import discover_data_files, list_data_files, read_problem_file, write_text
from tools.llm_client import build_llm_client
from tools.model_ids import normalize_model_ids
from tools.logging_setup import setup_logging
from tools.rework_router import build_rework_plan, write_rework_plan


# ── phases that pause for user confirmation ──
_CHECKPOINT_PHASES: set[WorkflowPhase] = {
    WorkflowPhase.MODEL_DECISION,
    WorkflowPhase.EXPERIMENT_PLAN,
    WorkflowPhase.CODE_PLAN,
    WorkflowPhase.RESULT_ANALYSIS,
    WorkflowPhase.PAPER_OUTLINE,
    WorkflowPhase.LANGUAGE_REVIEW,
}

# ── dependency invalidation: phase → downstream phases to invalidate ──
_INVALIDATION_MAP: dict[WorkflowPhase, list[WorkflowPhase]] = {
    WorkflowPhase.PROBLEM_ANALYSIS: [
        WorkflowPhase.MODEL_PROPOSAL, WorkflowPhase.MODEL_CRITIQUE, WorkflowPhase.MODEL_DECISION,
        WorkflowPhase.EXPERIMENT_PLAN, WorkflowPhase.CODE_PLAN, WorkflowPhase.CODE_GENERATION,
        WorkflowPhase.EXECUTION, WorkflowPhase.RESULT_ANALYSIS, WorkflowPhase.EVIDENCE_MAPPING,
        WorkflowPhase.PAPER_OUTLINE, WorkflowPhase.SECTION_WRITING,
        WorkflowPhase.FACT_REVIEW, WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW, WorkflowPhase.LANGUAGE_REVIEW,
    ],
    WorkflowPhase.MODEL_DECISION: [
        WorkflowPhase.EXPERIMENT_PLAN, WorkflowPhase.CODE_PLAN, WorkflowPhase.CODE_GENERATION,
        WorkflowPhase.EXECUTION, WorkflowPhase.RESULT_ANALYSIS, WorkflowPhase.EVIDENCE_MAPPING,
        WorkflowPhase.PAPER_OUTLINE, WorkflowPhase.SECTION_WRITING,
        WorkflowPhase.FACT_REVIEW, WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW, WorkflowPhase.LANGUAGE_REVIEW,
    ],
    WorkflowPhase.EXPERIMENT_PLAN: [
        WorkflowPhase.CODE_PLAN, WorkflowPhase.CODE_GENERATION,
        WorkflowPhase.EXECUTION, WorkflowPhase.RESULT_ANALYSIS, WorkflowPhase.EVIDENCE_MAPPING,
        WorkflowPhase.PAPER_OUTLINE, WorkflowPhase.SECTION_WRITING,
    ],
    WorkflowPhase.EXECUTION: [
        WorkflowPhase.RESULT_ANALYSIS, WorkflowPhase.EVIDENCE_MAPPING,
        WorkflowPhase.PAPER_OUTLINE, WorkflowPhase.SECTION_WRITING,
        WorkflowPhase.FACT_REVIEW, WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW, WorkflowPhase.LANGUAGE_REVIEW,
    ],
    WorkflowPhase.RESULT_ANALYSIS: [
        WorkflowPhase.EVIDENCE_MAPPING,
        WorkflowPhase.PAPER_OUTLINE, WorkflowPhase.SECTION_WRITING,
        WorkflowPhase.FACT_REVIEW,
        WorkflowPhase.LANGUAGE_REVIEW,
    ],
    WorkflowPhase.PAPER_OUTLINE: [
        WorkflowPhase.SECTION_WRITING,
        WorkflowPhase.FACT_REVIEW, WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW, WorkflowPhase.LANGUAGE_REVIEW,
    ],
    WorkflowPhase.SECTION_WRITING: [
        WorkflowPhase.FACT_REVIEW, WorkflowPhase.MATH_REVIEW,
        WorkflowPhase.STRUCTURE_REVIEW, WorkflowPhase.LANGUAGE_REVIEW,
    ],
}


def _derive_phase_for_agent(agent_name: str) -> WorkflowPhase:
    """Map agent names to workflow phases."""
    _MAP: dict[str, WorkflowPhase] = {
        "problem_agent": WorkflowPhase.PROBLEM_ANALYSIS,
        "modeling_agent": WorkflowPhase.MODEL_PROPOSAL,
        "modeling_critic_agent": WorkflowPhase.MODEL_CRITIQUE,
        "model_selection_agent": WorkflowPhase.MODEL_DECISION,
        "decision_agent": WorkflowPhase.MODEL_DECISION,
        "formulation_agent": WorkflowPhase.EXPERIMENT_PLAN,
        "experiment_plan_agent": WorkflowPhase.EXPERIMENT_PLAN,
        "code_plan_agent": WorkflowPhase.CODE_PLAN,
        "coding_agent": WorkflowPhase.CODE_GENERATION,
        "execution_agent": WorkflowPhase.EXECUTION,
        "analysis_agent": WorkflowPhase.RESULT_ANALYSIS,
        "evidence_agent": WorkflowPhase.EVIDENCE_MAPPING,
        "outline_agent": WorkflowPhase.PAPER_OUTLINE,
        "writing_agent": WorkflowPhase.SECTION_WRITING,
        "math_reviewer": WorkflowPhase.MATH_REVIEW,
        "fact_reviewer": WorkflowPhase.FACT_REVIEW,
        "structure_reviewer": WorkflowPhase.STRUCTURE_REVIEW,
        "language_reviewer": WorkflowPhase.LANGUAGE_REVIEW,
        "review_agent": WorkflowPhase.LANGUAGE_REVIEW,
        "export_agent": WorkflowPhase.EXPORT,
    }
    return _MAP.get(agent_name, WorkflowPhase.COMPLETE)


class ModelingWorkflow:
    def __init__(
        self,
        use_llm: bool = False,
        export_formats: list[str] | None = None,
        skip_review: bool = False,
        skip_export: bool = False,
        workspace: WorkspaceConfig | Path | None = None,
        run_workspace: bool = False,
        run_id: str | None = None,
        progress_callback: Callable[[str, str, WorkflowState | None], None] | None = None,
        pause_callback: Callable[[WorkflowPhase, WorkflowState], bool] | None = None,
    ) -> None:
        self.llm = build_llm_client(use_llm)
        self.skip_review = skip_review
        self.skip_export = skip_export
        self.workspace = self._resolve_workspace(workspace, run_workspace, run_id)
        self.progress_callback = progress_callback
        self.pause_callback = pause_callback
        # legacy flat agent list — kept for backward compat
        self.agents = [
            ProblemAgent(),
            ModelingAgent(),
            ModelingCriticAgent(),
            ModelSelectionAgent(),
            DecisionAgent(),
            FormulationAgent(),
            ExperimentPlanAgent(),
            CodePlanAgent(),
            CodingAgent(),
            ExecutionAgent(),
            AnalysisAgent(),
            EvidenceAgent(),
            OutlineAgent(),
            WritingAgent(),
            MathReviewerAgent(),
            FactReviewerAgent(),
            StructureReviewerAgent(),
            LanguageReviewerAgent(),
            ReviewAgent(),
        ]
        if export_formats and not skip_export:
            self.agents.append(ExportAgent(formats=export_formats))
        # phase → agent(s) mapping for staged execution
        self._phase_agent_map: dict[WorkflowPhase, list[Any]] = {}  # built lazily via property
        self._phase_agent_map_cache: dict[WorkflowPhase, list[Any]] = {}
        # stateful execution
        self._state: WorkflowState | None = None
        self._paused_at: WorkflowPhase | None = None

    # ── phase-agent index ────────────────────────────────────────────────
    def _build_phase_map(self) -> dict[WorkflowPhase, list[Any]]:
        mapping: dict[WorkflowPhase, list[Any]] = {}
        for agent in self.agents:
            phase = _derive_phase_for_agent(agent.name)
            mapping.setdefault(phase, []).append(agent)
        return mapping

    @property
    def phase_agent_map(self) -> dict[WorkflowPhase, list[Any]]:
        """Dynamic phase→agents map; cached per call to avoid rebuilds."""
        if not self._phase_agent_map_cache:
            self._phase_agent_map_cache = self._build_phase_map()
        return self._phase_agent_map_cache

    def _invalidate_phase_cache(self) -> None:
        self._phase_agent_map_cache = {}

    def _agents_for_phase(self, phase: WorkflowPhase) -> list[Any]:
        return self.phase_agent_map.get(phase, [])

    # ── workspace resolution ─────────────────────────────────────────────
    @staticmethod
    def _resolve_workspace(
        workspace: WorkspaceConfig | Path | None,
        run_workspace: bool,
        run_id: str | None,
    ) -> WorkspaceConfig:
        if workspace is not None:
            if run_workspace or run_id:
                raise ValueError("workspace cannot be combined with run_workspace or run_id")
            if isinstance(workspace, WorkspaceConfig):
                return workspace
            return WorkspaceConfig.from_root(Path(workspace), project_root=PROJECT_ROOT)
        if run_workspace or run_id:
            return WorkspaceConfig.isolated_run(PROJECT_ROOT, run_id)
        return WORKSPACE

    # ── state initialisation ─────────────────────────────────────────────
    def _init_state(self, problem_text: str, data_files: list[Path] | None = None) -> WorkflowState:
        self.workspace.ensure_dirs()
        setup_logging(self.workspace.logs_dir)
        self.llm.set_log_path(self.workspace.logs_dir / "llm_calls.jsonl")

        selected_data_files = list_data_files(data_files or [])
        if not selected_data_files:
            selected_data_files = discover_data_files(self.workspace.data_dir)
        if not selected_data_files and self.workspace.root != WORKSPACE.root:
            selected_data_files = discover_data_files(WORKSPACE.data_dir)

        state = WorkflowState(
            problem_text=problem_text,
            data_files=selected_data_files,
            workspace=self.workspace,
            llm=self.llm,
        )
        state.notes[K_LLM_STATUS] = "enabled" if self.llm.enabled else self.llm.config.reason
        state.notes["workspace_root"] = str(self.workspace.root)
        state.phase = WorkflowPhase.PROBLEM_ANALYSIS
        write_text(self.workspace.input_dir / "problem.txt", problem_text)
        return state

    # ── legacy fire-and-forget run ───────────────────────────────────────
    def run(self, problem_text: str, data_files: list[Path] | None = None) -> WorkflowState:
        """Legacy one-shot execution: runs all phases without pausing."""
        return self.run_until(WorkflowPhase.COMPLETE, problem_text, data_files, auto_approve=True)

    # ── staged execution ─────────────────────────────────────────────────
    def run_until(
        self,
        target_phase: WorkflowPhase,
        problem_text: str | None = None,
        data_files: list[Path] | None = None,
        auto_approve: bool = False,
    ) -> WorkflowState:
        """Execute phases from current position up to *target_phase*.

        Pauses at checkpoint phases unless *auto_approve* is True.
        Call ``approve()`` then ``resume()`` to continue after a pause.
        """
        log = logging.getLogger("mma.workflow")

        # initialise or reuse state
        if self._state is None and problem_text is not None:
            self._state = self._init_state(problem_text, data_files)
        elif self._state is None:
            raise RuntimeError("No existing state; provide problem_text on first call.")

        state = self._state

        # determine start phase
        if self._paused_at is not None:
            # if the paused phase is already approved, start from next
            paused_status = state.get_phase_status(self._paused_at)
            if paused_status in (PhaseStatus.APPROVED, PhaseStatus.COMPLETED):
                next_p = state.next_pending_phase()
                start_phase = next_p if next_p else self._paused_at
            else:
                start_phase = self._paused_at
            self._paused_at = None
        else:
            start_phase = state.phase

        phases = [p for p in WorkflowPhase if p.order >= start_phase.order and p.order <= target_phase.order]
        if not phases:
            return state

        for phase in phases:
            current_status = state.get_phase_status(phase)
            if current_status in (PhaseStatus.COMPLETED, PhaseStatus.APPROVED):
                continue

            state.phase = phase
            state.set_phase_status(phase, PhaseStatus.RUNNING)
            agents = self._agents_for_phase(phase)

            if not agents:
                state.set_phase_status(phase, PhaseStatus.COMPLETED)
                continue

            for agent in agents:
                skip_reason = self._should_skip_agent(agent.name, state)
                if skip_reason:
                    log.info("Skipping %s: %s", agent.name, skip_reason)
                    self._emit_progress(agent.name, "skipped", state)
                    continue

                self._emit_progress(agent.name, "started", state)
                t0 = time.perf_counter()
                progress_status = "completed"
                try:
                    state = agent.run(state)
                    self._state = state
                except Exception as exc:
                    error = state.add_error(agent.name, exc, recoverable=False)
                    progress_status = "failed"
                    log.exception(
                        "Agent %s crashed [%s/%s]: %s",
                        agent.name,
                        error["category"],
                        error["exception_type"],
                        error["message"],
                    )
                state.set_duration(agent.name, time.perf_counter() - t0)
                log.info("%s done in %.2fs", agent.name, state.durations[agent.name])
                self._emit_progress(agent.name, progress_status, state)

            state.set_phase_status(phase, PhaseStatus.COMPLETED)

            # checkpoint: pause for user confirmation
            if phase in _CHECKPOINT_PHASES and not auto_approve:
                state.set_phase_status(phase, PhaseStatus.WAITING_FOR_USER)
                self._paused_at = phase
                # call pause callback if registered; if it returns False, don't auto-resume
                if self.pause_callback:
                    try:
                        accepted = self.pause_callback(phase, state)
                        if accepted:
                            state.set_phase_status(phase, PhaseStatus.APPROVED)
                            self._paused_at = None
                            state.record_decision(phase, "approved", operator="auto")
                            continue
                    except Exception:
                        logging.getLogger("mma.workflow").debug("Pause callback failed.", exc_info=True)
                # save state for resumption
                self._save_pause_state(state)
                return state

            state.set_phase_status(phase, PhaseStatus.COMPLETED)

        # post-execution: writing-review retry loop (for legacy compat)
        if target_phase in (WorkflowPhase.COMPLETE, WorkflowPhase.LANGUAGE_REVIEW):
            state = self._retry_writing_review(state, log)

        # write diagnostics
        self._write_diagnostics(state, log)
        return state

    def approve(self, phase: WorkflowPhase, edits: dict[str, Any] | None = None) -> WorkflowState:
        """Approve a paused phase, optionally injecting edits into state."""
        if self._state is None:
            raise RuntimeError("No workflow state to approve.")
        state = self._state

        if edits:
            self._apply_edits(state, phase, edits)

        state.set_phase_status(phase, PhaseStatus.APPROVED)
        state.record_decision(phase, "approved", operator="user", notes=str(edits or {}))
        # keep _paused_at so resume() knows where to continue from
        self._clear_pause_state()
        return state

    def resume(self) -> WorkflowState:
        """Continue execution from the last paused/approved phase."""
        if self._state is None:
            raise RuntimeError("No workflow state to resume.")
        # find the next incomplete phase
        next_phase = self._state.next_pending_phase()
        if next_phase is None or next_phase == WorkflowPhase.COMPLETE:
            return self._state
        self._paused_at = None
        return self.run_until(WorkflowPhase.COMPLETE, auto_approve=False)

    def rerun_from(self, phase: WorkflowPhase) -> WorkflowState:
        """Invalidate *phase* and all downstream phases, then re-execute from *phase*."""
        if self._state is None:
            raise RuntimeError("No workflow state to rerun.")
        state = self._state
        log = logging.getLogger("mma.workflow")

        # invalidate the target phase
        state.set_phase_status(phase, PhaseStatus.NEEDS_REVISION)
        state.record_decision(phase, "rerun", operator="system", notes=f"Rerun from {phase.value}")

        # cascade invalidation to downstream
        downstream = _INVALIDATION_MAP.get(phase, [])
        for down_phase in downstream:
            state.set_phase_status(down_phase, PhaseStatus.NEEDS_REVISION)
            log.info("Invalidated downstream phase: %s", down_phase.value)

        self._paused_at = None
        return self.run_until(WorkflowPhase.COMPLETE, auto_approve=False)

    def revise_artifact(self, artifact_name: str, feedback: str) -> WorkflowState:
        """Locally revise a single artifact based on feedback.

        Supported artifact names:
        - ``paper_outline`` — regenerate outline only
        - ``paper_section:{id}`` — rewrite one section (e.g. ``paper_section:abstract``)
        - ``paper_draft`` — rewrite full paper
        - ``review_finding:{id}`` — address one review issue
        - ``model_decision`` — update model selection
        """
        if self._state is None:
            raise RuntimeError("No workflow state to revise.")
        state = self._state
        log = logging.getLogger("mma.workflow")

        if artifact_name == "paper_outline":
            state.record_decision(WorkflowPhase.PAPER_OUTLINE, "revise", notes=feedback)
            state.set_phase_status(WorkflowPhase.PAPER_OUTLINE, PhaseStatus.NEEDS_REVISION)
            state.set_phase_status(WorkflowPhase.SECTION_WRITING, PhaseStatus.NEEDS_REVISION)
            for agent in self.agents:
                if agent.name == "writing_agent":
                    self._emit_progress("writing_agent_revise_outline", "started", state)
                    try:
                        state = agent.run(state)
                        self._state = state
                    except Exception as exc:
                        state.add_error("writing_agent_revise_outline", exc, recoverable=False)
                        log.exception("Revise outline failed: %s", exc)
                    self._emit_progress("writing_agent_revise_outline", "completed", state)
                    break
            state.set_phase_status(WorkflowPhase.PAPER_OUTLINE, PhaseStatus.COMPLETED)
            state.set_phase_status(WorkflowPhase.SECTION_WRITING, PhaseStatus.COMPLETED)

        elif artifact_name.startswith("paper_section:"):
            section_id = artifact_name.split(":", 1)[1]
            state.record_decision(WorkflowPhase.SECTION_WRITING, "revise_section", notes=f"{section_id}: {feedback}")
            for agent in self.agents:
                if agent.name == "writing_agent":
                    self._emit_progress(f"writing_agent_revise_{section_id}", "started", state)
                    try:
                        # rebuild outline and rewrite just that section
                        outline = agent._generate_outline(state)
                        section_data = next((s for s in outline.sections if s.get("id") == section_id), None)
                        if section_data:
                            section_text = agent._write_section(state, section_data)
                            state.notes[f"revised_section_{section_id}"] = section_text
                            # re-assemble paper with revised section
                            current_sections = {s.get("id", ""): state.notes.get(f"revised_section_{s.get('id','')}", "") or "" for s in outline.sections}
                            current_sections[section_id] = section_text
                            paper = agent._assemble_and_unify(state, current_sections)
                            from tools.file_tool import write_text as _wt
                            from agents.base import A_PAPER
                            paper_path = _wt(state.workspace.paper_dir / "paper_draft.md", paper)
                            state.artifacts[A_PAPER] = paper_path
                    except Exception as exc:
                        state.add_error(f"writing_agent_revise_{section_id}", exc, recoverable=False)
                        log.exception("Revise section %s failed: %s", section_id, exc)
                    self._emit_progress(f"writing_agent_revise_{section_id}", "completed", state)
                    break

        elif artifact_name.startswith("review_finding:"):
            finding_id = artifact_name.split(":", 1)[1]
            state.record_decision(WorkflowPhase.LANGUAGE_REVIEW, "address_finding", notes=f"{finding_id}: {feedback}")
            # re-run review agents with the feedback injected
            state.notes["review_feedback_target"] = finding_id
            state.notes["review_feedback_text"] = feedback
            # re-run language reviewer to check the fix
            from agents.language_reviewer import LanguageReviewerAgent
            try:
                lr = LanguageReviewerAgent()
                state = lr.run(state)
                self._state = state
            except Exception as exc:
                state.add_error("language_reviewer_recheck", exc, recoverable=False)
                log.exception("Re-check after finding fix failed: %s", exc)

        elif artifact_name == "paper_draft":
            state.record_decision(WorkflowPhase.SECTION_WRITING, "revise", notes=feedback)
            state.set_phase_status(WorkflowPhase.SECTION_WRITING, PhaseStatus.NEEDS_REVISION)
            for agent in self.agents:
                if agent.name == "writing_agent":
                    self._emit_progress("writing_agent_revise_draft", "started", state)
                    try:
                        state = agent.run(state)
                        self._state = state
                    except Exception as exc:
                        state.add_error("writing_agent_revise_draft", exc, recoverable=False)
                        log.exception("Revise draft failed: %s", exc)
                    self._emit_progress("writing_agent_revise_draft", "completed", state)
                    break
            state.set_phase_status(WorkflowPhase.SECTION_WRITING, PhaseStatus.COMPLETED)

        elif artifact_name == "model_decision":
            state.record_decision(WorkflowPhase.MODEL_DECISION, "revise", notes=feedback)
            return self.rerun_from(WorkflowPhase.MODEL_DECISION)

        else:
            log.warning("Unknown artifact for revision: %s", artifact_name)

        self._write_diagnostics(state, log)
        return state

    # ── helpers ──────────────────────────────────────────────────────────
    def _should_skip_agent(self, agent_name: str, state: WorkflowState) -> str:
        """Return a reason string if agent should be skipped, empty string otherwise."""
        if agent_name in ("writing_agent", "review_agent"):
            if state.notes.get(K_EXECUTION_STATUS) == "failed":
                return "execution failed"
        if agent_name == "export_agent" and self.skip_export:
            return "skip-export flag"
        if agent_name == "review_agent" and self.skip_review:
            return "skip-review flag"
        return ""

    def _apply_edits(self, state: WorkflowState, phase: WorkflowPhase, edits: dict[str, Any]) -> None:
        """Inject user edits into the state based on the phase."""
        if phase == WorkflowPhase.MODEL_DECISION:
            if "primary_model_id" in edits:
                if state.model_decision is None:
                    from agents.base import ModelDecision
                    state.model_decision = ModelDecision()
                state.model_decision.primary_model_id = edits["primary_model_id"]
            if "selected_model_ids" in edits:
                if state.model_decision is None:
                    from agents.base import ModelDecision
                    state.model_decision = ModelDecision()
                normalized = normalize_model_ids(edits["selected_model_ids"])
                state.model_decision.selected_model_ids = normalized.selected
                if normalized.dropped:
                    state.notes["user_edit_dropped_model_ids"] = json.dumps(normalized.dropped, ensure_ascii=False)
                state.notes["selected_model_ids"] = json.dumps(normalized.selected)
            if "baseline_model_id" in edits:
                if state.model_decision is None:
                    from agents.base import ModelDecision
                    state.model_decision = ModelDecision()
                state.model_decision.baseline_model_id = edits["baseline_model_id"]
        elif phase == WorkflowPhase.EXPERIMENT_PLAN:
            if "metrics" in edits and state.experiment_plan is not None:
                state.experiment_plan.metrics = edits["metrics"]
        elif phase == WorkflowPhase.PAPER_OUTLINE:
            if "sections" in edits and state.paper_outline is not None:
                state.paper_outline.sections = edits["sections"]
        # generic fallback: inject into notes
        for key, value in edits.items():
            if isinstance(value, str):
                state.notes[f"user_edit_{key}"] = value

    def _save_pause_state(self, state: WorkflowState) -> None:
        """Persist state so the UI can reload it."""
        try:
            pause_path = self.workspace.logs_dir / "workflow_pause.json"
            pause_path.write_text(
                json.dumps(state.to_json(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logging.getLogger("mma.workflow").debug("Failed to save pause state.", exc_info=True)

    def _clear_pause_state(self) -> None:
        try:
            pause_path = self.workspace.logs_dir / "workflow_pause.json"
            if pause_path.exists():
                pause_path.unlink()
        except Exception:
            pass

    def _write_diagnostics(self, state: WorkflowState, log) -> None:
        rework_plan = build_rework_plan(state)
        if rework_plan is not None:
            route = rework_plan.route
            state.notes["auto_rework_target_phase"] = route.target_phase.value
            state.notes["auto_rework_reason"] = route.reason
            state.notes["auto_rework_severity"] = route.severity
            state.notes["auto_rework_rerun_from_phase"] = rework_plan.rerun_from_phase.value
            state.artifacts["auto_rework_plan"] = write_rework_plan(self.workspace, rework_plan)
        diag = {
            "errors": state.errors,
            "durations_s": state.durations,
            "execution_status": state.notes.get(K_EXECUTION_STATUS, "unknown"),
            "phase": state.phase.value,
            "phase_status": dict(state.phase_status),
            "decisions": state.decisions,
            "auto_rework_route": rework_plan.route.to_dict() if rework_plan else None,
            "auto_rework_plan": rework_plan.to_dict() if rework_plan else None,
        }
        diag["workspace_root"] = str(self.workspace.root)
        diag_path = self.workspace.logs_dir / "workflow_diagnostics.json"
        diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Workflow complete. Diagnostics written to %s", diag_path)

    def _emit_progress(self, agent_name: str, status: str, state: WorkflowState | None = None) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(agent_name, status, state)
        except Exception:
            logging.getLogger("mma.workflow").debug("Progress callback failed.", exc_info=True)

    def _retry_writing_review(self, state: WorkflowState, log) -> WorkflowState:
        """Retry WritingAgent → ReviewAgent up to 2 additional times if quality < 82."""
        if state.notes.get(K_EXECUTION_STATUS) == "failed":
            return state
        if self.skip_review:
            return state

        max_total_attempts = 3
        attempt = 1

        writing_agent = None
        review_agent = None
        for a in self.agents:
            if a.name == "writing_agent":
                writing_agent = a
            elif a.name == "review_agent":
                review_agent = a

        if writing_agent is None or review_agent is None:
            return state

        while attempt < max_total_attempts:
            score_str = state.notes.get(K_PAPER_QUALITY_SCORE, "0")
            try:
                score = int(float(score_str))
            except (ValueError, TypeError):
                score = 0

            if score >= 82:
                break

            stop_reason = self._writing_retry_stop_reason(state)
            if stop_reason:
                log.info("Skip WritingAgent retry: %s", stop_reason)
                state.notes["writing_retry_stop_reason"] = stop_reason
                break

            log.info(
                "Paper quality %d < 82 — retry %d/%d (WritingAgent → ReviewAgent)",
                score,
                attempt + 1,
                max_total_attempts,
            )

            t0 = time.perf_counter()
            writing_retry_name = f"writing_agent_retry_{attempt + 1}"
            self._emit_progress(writing_retry_name, "started", state)
            try:
                state = writing_agent.run(state)
            except Exception as exc:
                error = state.add_error(writing_retry_name, exc, recoverable=False)
                log.exception(
                    "WritingAgent retry %d crashed [%s/%s]: %s",
                    attempt + 1,
                    error["category"],
                    error["exception_type"],
                    error["message"],
                )
                self._emit_progress(writing_retry_name, "failed", state)
                break
            state.set_duration(writing_retry_name, time.perf_counter() - t0)
            log.info("WritingAgent retry %d done in %.2fs", attempt + 1, state.durations[writing_retry_name])
            self._emit_progress(writing_retry_name, "completed", state)

            t0 = time.perf_counter()
            review_retry_name = f"review_agent_retry_{attempt + 1}"
            self._emit_progress(review_retry_name, "started", state)
            try:
                state = review_agent.run(state)
            except Exception as exc:
                error = state.add_error(review_retry_name, exc, recoverable=False)
                log.exception(
                    "ReviewAgent retry %d crashed [%s/%s]: %s",
                    attempt + 1,
                    error["category"],
                    error["exception_type"],
                    error["message"],
                )
                self._emit_progress(review_retry_name, "failed", state)
                break
            state.set_duration(review_retry_name, time.perf_counter() - t0)
            log.info("ReviewAgent retry %d done in %.2fs", attempt + 1, state.durations[review_retry_name])
            self._emit_progress(review_retry_name, "completed", state)

            attempt += 1

        return state

    def _writing_retry_stop_reason(self, state: WorkflowState) -> str:
        failure_kind = state.notes.get(K_LLM_FAILURE_KIND, "")
        if failure_kind in {"quota", "auth", "billing"}:
            return f"non-retryable LLM failure: {failure_kind}"
        if state.notes.get(K_PREWRITING_GATE_STATUS) == "blocked":
            return "pre-writing gate blocked missing core evidence"
        for key, value in state.notes.items():
            if key.endswith("_error") and _looks_like_non_retryable_llm_error(str(value)):
                return "non-retryable LLM error"
        return ""


def _looks_like_non_retryable_llm_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient balance",
            "quota",
            "billing",
            "402",
            "unauthorized",
            "invalid api key",
        )
    )


def run_from_files(
    problem_file: Path,
    data_files: list[Path],
    use_llm: bool = False,
    export_formats: list[str] | None = None,
    workspace: WorkspaceConfig | Path | None = None,
    run_workspace: bool = False,
    run_id: str | None = None,
) -> WorkflowState:
    workflow = ModelingWorkflow(
        use_llm=use_llm,
        export_formats=export_formats,
        workspace=workspace,
        run_workspace=run_workspace,
        run_id=run_id,
    )
    return workflow.run(read_problem_file(problem_file), data_files)
