from __future__ import annotations

import json

from agents.base import A_FORMULATION_SPEC, Agent, WorkflowState
from tools.file_tool import write_text
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
