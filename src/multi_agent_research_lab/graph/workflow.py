"""LangGraph workflow for orchestrating multi-agent research."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with LangGraph."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._graph: Any = self.build()

    def build(self) -> Any:
        """Create and compile the LangGraph workflow graph."""
        builder: StateGraph[Any, Any, Any, Any] = StateGraph(ResearchState)

        # 1. Define nodes
        def node_supervisor(state: ResearchState) -> ResearchState:
            with trace_span("node_supervisor"):
                return self.supervisor.run(state)

        def node_researcher(state: ResearchState) -> ResearchState:
            with trace_span("node_researcher"):
                return self.researcher.run(state)

        def node_analyst(state: ResearchState) -> ResearchState:
            with trace_span("node_analyst"):
                return self.analyst.run(state)

        def node_writer(state: ResearchState) -> ResearchState:
            with trace_span("node_writer"):
                return self.writer.run(state)

        def node_critic(state: ResearchState) -> ResearchState:
            with trace_span("node_critic"):
                return self.critic.run(state)

        builder.add_node("supervisor", node_supervisor)
        builder.add_node("researcher", node_researcher)
        builder.add_node("analyst", node_analyst)
        builder.add_node("writer", node_writer)
        builder.add_node("critic", node_critic)

        # 2. Set entry point
        builder.set_entry_point("supervisor")

        # 3. Routing decision function
        def route_decision(state: ResearchState) -> str:
            last_route = state.route_history[-1] if state.route_history else "done"
            if last_route in {"researcher", "analyst", "writer", "critic"}:
                return last_route
            return END

        builder.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # 4. Return worker outputs back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the updated ResearchState."""
        with trace_span("multi_agent_workflow", attributes={"query": state.request.query}) as span:
            try:
                output = self._graph.invoke(state)
                if isinstance(output, dict):
                    final_state = ResearchState.model_validate(output)
                elif isinstance(output, ResearchState):
                    final_state = output
                else:
                    final_state = state
                span["status"] = "success"
                return final_state
            except Exception as exc:
                logger.exception("Multi-agent workflow execution failed: %s", exc)
                state.add_error(f"Workflow error: {exc}")
                span["status"] = "error"
                span["error"] = str(exc)
                return state
