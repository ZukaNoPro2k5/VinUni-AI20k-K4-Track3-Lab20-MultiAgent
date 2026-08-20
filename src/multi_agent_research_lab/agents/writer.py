"""Writer agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "N/A"
        analysis_notes = state.analysis_notes or "N/A"

        sources_summary = "\n".join(
            f"- [{doc.metadata.get('source_id', f'Doc {idx + 1}')}] "
            f"{doc.title} ({doc.url or 'local'})"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Principal Technical Writer. Synthesize the provided research and analysis "
            "notes into a comprehensive, beautifully structured technical report. Ensure claims "
            "are grounded in evidence with inline citations and a formal references section."
        )
        user_prompt = (
            f"Research Question: {query}\n"
            f"Target Audience: {audience}\n\n"
            f"Sources Available:\n{sources_summary}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            "Please compose a complete, authoritative Markdown report with:\n"
            "1. Executive Summary\n"
            "2. In-Depth Technical Analysis & Architecture\n"
            "3. Comparative Trade-offs & Production Guardrails\n"
            "4. Conclusions & Recommendations\n"
            "5. References with Citations"
        )

        response = self.llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content

        state.add_trace_event(
            "writing_complete",
            {
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
                "cost_usd": response.cost_usd,
            },
        )
        state.add_agent_result(
            agent=AgentName.WRITER,
            content=state.final_answer,
            metadata={"cost_usd": response.cost_usd},
        )

        logger.info("Writer generated final answer")
        return state
