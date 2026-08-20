"""Critic / verifier agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking, verification, and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append verification findings."""
        query = state.request.query
        final_answer = state.final_answer or "No final answer drafted."
        sources_summary = "\n".join(f"- {s.title}" for s in state.sources)

        system_prompt = (
            "You are a Rigorous Fact-Checker and Critic. Your task is to evaluate the drafted "
            "report for factual consistency, citation coverage, completeness, and clarity."
        )
        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Verified Sources:\n{sources_summary}\n\n"
            f"Drafted Final Answer:\n{final_answer}\n\n"
            "Evaluate the draft and provide:\n"
            "- Citation coverage & evidence grounding score (0-10)\n"
            "- Fact-check verdict (Passed / Needs Revision)\n"
            "- Key strengths and any potential hallucination risks"
        )

        response = self.llm.complete(system_prompt, user_prompt)
        state.critic_notes = response.content

        state.add_trace_event(
            "critic_complete",
            {
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
                "cost_usd": response.cost_usd,
            },
        )
        state.add_agent_result(
            agent=AgentName.CRITIC,
            content=state.critic_notes,
            metadata={"cost_usd": response.cost_usd},
        )

        logger.info("Critic completed verification review")
        return state
