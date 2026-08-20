"""Analyst agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights and critical analysis."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        query = state.request.query
        notes = state.research_notes or "No research notes provided."

        system_prompt = (
            "You are a Senior Systems Analyst. Your goal is to critically evaluate raw findings, "
            "synthesize comparative trade-offs (latency, cost, accuracy), identify risks, "
            "and structure actionable insights."
        )
        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Research Notes:\n{notes}\n\n"
            "Please analyze these findings. Provide:\n"
            "1. Core Mechanisms and Architectural Comparisons\n"
            "2. Trade-offs (e.g. latency vs cost vs quality)\n"
            "3. Potential failure modes and operational risks\n"
            "4. Synthesis of key claims and supporting evidence strength"
        )

        response = self.llm.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content

        state.add_trace_event(
            "analysis_complete",
            {
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
                "cost_usd": response.cost_usd,
            },
        )
        state.add_agent_result(
            agent=AgentName.ANALYST,
            content=state.analysis_notes,
            metadata={"cost_usd": response.cost_usd},
        )

        logger.info("Analyst created analytical notes")
        return state
