"""Unit tests for LangGraph workflow orchestration."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_end_to_end() -> None:
    workflow = MultiAgentWorkflow()
    initial_state = ResearchState(
        request=ResearchQuery(query="Research GraphRAG state-of-the-art and production guardrails")
    )
    result = workflow.run(initial_state)

    assert result.final_answer is not None
    assert len(result.sources) > 0
    assert result.research_notes is not None
    assert result.analysis_notes is not None
    assert result.critic_notes is not None
    assert "researcher" in result.route_history
    assert "analyst" in result.route_history
    assert "writer" in result.route_history
    assert "critic" in result.route_history
    assert result.route_history[-1] == "done"
