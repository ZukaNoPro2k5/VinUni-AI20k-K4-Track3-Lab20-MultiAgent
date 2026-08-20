"""Unit tests for LLM and Search services."""

from __future__ import annotations

from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_client_local_completion() -> None:
    client = LLMClient()
    response = client.complete("You are a technical writer", "Summarize findings")
    assert response.content
    assert response.input_tokens is not None
    assert response.output_tokens is not None
    assert response.cost_usd is not None


def test_search_client_offline_corpus() -> None:
    client = SearchClient()
    results = client.search("GraphRAG architecture and multi-agent", max_results=3)
    assert len(results) > 0
    assert results[0].title
    assert results[0].snippet
