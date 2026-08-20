"""Shared state for the multi-agent workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None
    critic_notes: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_agent_result(
        self, agent: AgentName | str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        agent_name = (
            AgentName(agent)
            if isinstance(agent, str) and agent in AgentName._value2member_map_
            else AgentName.SUPERVISOR
        )
        self.agent_results.append(
            AgentResult(agent=agent_name, content=content, metadata=metadata or {})
        )

    def add_error(self, error: str) -> None:
        self.errors.append(error)
