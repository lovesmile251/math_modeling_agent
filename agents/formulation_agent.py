from __future__ import annotations

import json

from agents.base import A_FORMULATION_SPEC, Agent, WorkflowState
from tools.file_tool import write_text
from tools.model_ids import normalize_model_decision
from tools.modeling_dsl import build_formulation


class FormulationAgent(Agent):
    """Translate the structured problem into an auditable modeling DSL."""

    name = "formulation_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        selected = (
            state.model_decision.selected_model_ids
            if state.model_decision
            else []
        )
        if state.model_decision:
            normalized = normalize_model_decision(
                selected_model_ids=state.model_decision.selected_model_ids,
                primary_model_id=state.model_decision.primary_model_id,
                baseline_model_id=state.model_decision.baseline_model_id,
            )
            state.model_decision.selected_model_ids = normalized.selected
            state.model_decision.primary_model_id = normalized.primary
            state.model_decision.baseline_model_id = normalized.baseline
            selected = normalized.selected
            if normalized.dropped:
                state.notes["formulation_dropped_model_ids"] = ", ".join(normalized.dropped)
        formulation = build_formulation(state.problem_spec, selected)
        state.formulation_spec = formulation
        path = write_text(
            state.workspace.logs_dir / "formulation_spec.json",
            json.dumps(formulation.__dict__, ensure_ascii=False, indent=2),
        )
        state.artifacts[A_FORMULATION_SPEC] = path
        if formulation.validation_issues:
            state.notes["formulation_status"] = "needs_revision"
            state.notes["formulation_issues"] = "; ".join(formulation.validation_issues)
        else:
            state.notes["formulation_status"] = "valid"
        return state
