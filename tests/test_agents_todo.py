"""Unit tests for agents and supervisor routing policy."""

from __future__ import annotations

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routing_progression() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent architecture"))
    supervisor = SupervisorAgent()

    # Step 1: Initial state -> routes to researcher
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"
    assert state.iteration == 1

    # Step 2: Add sources and research notes -> routes to analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.research_notes = "Found key concepts on MAS."
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"
    assert state.iteration == 2

    # Step 3: Add analysis notes -> routes to writer
    state.analysis_notes = "Analyzed trade-offs."
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"
    assert state.iteration == 3

    # Step 4: Add final answer -> routes to critic
    state.final_answer = "Final report."
    state = supervisor.run(state)
    assert state.route_history[-1] == "critic"
    assert state.iteration == 4

    # Step 5: Add critic notes -> routes to done
    state.critic_notes = "Verified."
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_max_iterations_guardrail() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent architecture"))
    supervisor = SupervisorAgent()
    state.iteration = supervisor.max_iterations

    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_worker_agents_execution() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG state-of-the-art"))

    # Researcher
    researcher = ResearcherAgent()
    state = researcher.run(state)
    assert len(state.sources) > 0
    assert state.research_notes is not None
    assert any(res.agent == AgentName.RESEARCHER for res in state.agent_results)

    # Analyst
    analyst = AnalystAgent()
    state = analyst.run(state)
    assert state.analysis_notes is not None
    assert any(res.agent == AgentName.ANALYST for res in state.agent_results)

    # Writer
    writer = WriterAgent()
    state = writer.run(state)
    assert state.final_answer is not None
    assert any(res.agent == AgentName.WRITER for res in state.agent_results)

    # Critic
    critic = CriticAgent()
    state = critic.run(state)
    assert state.critic_notes is not None
    assert any(res.agent == AgentName.CRITIC for res in state.agent_results)
