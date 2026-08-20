"""Researcher agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm: LLMClient | None = None
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # 1. Search for sources
        docs = self.search_client.search(query, max_results=max_sources)

        # Deduplicate with any existing sources
        existing_urls = {s.url for s in state.sources if s.url}
        new_sources: list[SourceDocument] = list(state.sources)
        for doc in docs:
            if not doc.url or doc.url not in existing_urls:
                new_sources.append(doc)
                if doc.url:
                    existing_urls.add(doc.url)

        state.sources = new_sources

        # 2. Synthesize research notes using LLM
        sources_text = "\n\n".join(
            f"[{idx + 1}] Title: {doc.title}\nSource: {doc.url or 'N/A'}\nSnippet: {doc.snippet}"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = (
            "You are an expert Research Agent. Your job is to extract factual findings, "
            "data points, architectural patterns, and verified citations from the provided sources."
        )
        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Discovered Sources:\n{sources_text}\n\n"
            "Please produce structured research notes with key findings and source citations."
        )

        response = self.llm.complete(system_prompt, user_prompt)
        state.research_notes = response.content

        state.add_trace_event(
            "research_complete",
            {
                "query": query,
                "sources_found": len(state.sources),
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
                "cost_usd": response.cost_usd,
            },
        )
        state.add_agent_result(
            agent=AgentName.RESEARCHER,
            content=state.research_notes,
            metadata={"source_count": len(state.sources), "cost_usd": response.cost_usd},
        )

        logger.info("Researcher gathered %d sources and created research notes", len(state.sources))
        return state
