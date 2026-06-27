from __future__ import annotations

import logging

from agents.base import Agent, PaperOutline, WorkflowState

log = logging.getLogger("mma.outline_agent")


class OutlineAgent(Agent):
    """Generates the paper outline with evidence pre-assignment.

    Delegates to WritingAgent._generate_outline via lightweight coupling.
    Runs as the PAPER_OUTLINE phase, before SECTION_WRITING.
    """

    name = "outline_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        from agents.writing_agent import WritingAgent as WA
        # use a throwaway WritingAgent instance just for outline generation
        wa = WA()
        try:
            outline = wa._generate_outline(state)
            state.paper_outline = outline
            log.info("Outline generated: %d sections", outline.total_sections)
        except Exception as exc:
            log.warning("Outline generation failed: %s", exc)
            # fallback: minimal outline
            outline = PaperOutline(
                sections=[
                    {"id": "abstract", "title": "摘要", "available_claims": [], "available_figures": [], "available_tables": []},
                    {"id": "model", "title": "模型建立与求解", "available_claims": [], "available_figures": [], "available_tables": []},
                    {"id": "results", "title": "结果分析", "available_claims": [], "available_figures": [], "available_tables": []},
                    {"id": "conclusion", "title": "结论", "available_claims": [], "available_figures": [], "available_tables": []},
                ],
                total_sections=4,
            )
            state.paper_outline = outline
        return state
