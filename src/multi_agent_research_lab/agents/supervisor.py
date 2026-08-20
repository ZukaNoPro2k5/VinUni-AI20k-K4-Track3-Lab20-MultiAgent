"""Supervisor / router agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm: LLMClient | None = None, enable_critic: bool = True) -> None:
        self.llm = llm or LLMClient()
        self.enable_critic = enable_critic
        self.max_iterations = get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route and record decision."""
        # 1. Guardrail: Max iterations check
        if state.iteration >= self.max_iterations:
            decision = "done"
            reason = f"Max iterations ({self.max_iterations}) reached."
        # 2. Sequential pipeline logic with condition checks
        elif not state.sources or not state.research_notes:
            decision = "researcher"
            reason = "Missing sources or research notes."
        elif not state.analysis_notes:
            decision = "analyst"
            reason = "Research notes collected; analysis needed."
        elif not state.final_answer:
            decision = "writer"
            reason = "Analysis complete; synthesizing final answer."
        elif self.enable_critic and not state.critic_notes:
            decision = "critic"
            reason = "Final answer drafted; running validation."
        else:
            decision = "done"
            reason = "All pipeline stages completed."

        state.record_route(decision)
        state.add_trace_event(
            "supervisor_decision",
            {
                "decision": decision,
                "iteration": state.iteration,
                "reason": reason,
            },
        )
        state.add_agent_result(
            agent=AgentName.SUPERVISOR,
            content=f"Next route: {decision} (Reason: {reason})",
            metadata={"decision": decision, "iteration": state.iteration},
        )
        logger.info("Supervisor decided next step: %s (iteration %d)", decision, state.iteration)
        return state
