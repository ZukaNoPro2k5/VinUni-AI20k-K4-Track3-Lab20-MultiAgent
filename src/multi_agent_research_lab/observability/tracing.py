"""Tracing hooks and span manager for observability."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# In-memory trace span repository
_ACTIVE_SPANS: list[dict[str, Any]] = []


def setup_external_tracing() -> None:
    """Initialize LangSmith or Langfuse tracing environment variables if configured."""
    settings = get_settings()

    if settings.langsmith_api_key and settings.langsmith_api_key.strip() not in {
        "",
        "...",
        "your_langsmith_api_key",
    }:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager to record execution span latency, attributes, and errors."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "in_progress",
    }
    _ACTIVE_SPANS.append(span)
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug(
            "Span '%s' finished in %.4fs (status: %s)",
            name,
            span["duration_seconds"],
            span["status"],
        )


def get_recorded_spans() -> list[dict[str, Any]]:
    """Return all recorded trace spans."""
    return list(_ACTIVE_SPANS)


def clear_spans() -> None:
    """Clear recorded trace spans."""
    _ACTIVE_SPANS.clear()
