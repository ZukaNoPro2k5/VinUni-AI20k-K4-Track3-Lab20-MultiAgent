"""Benchmark runner for comparing single-agent vs multi-agent."""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def evaluate_quality(state: ResearchState) -> float:
    """Calculate quality score (0.0 - 10.0) based on structure, depth, citations, and clarity."""
    if not state.final_answer or state.errors:
        return 0.0

    score = 4.0  # Base for having a final answer
    text = state.final_answer

    # 1. Structural completeness (+2.0)
    if "##" in text:
        score += 1.0
    if re.search(r"(summary|overview|introduction)", text, re.IGNORECASE):
        score += 0.5
    if re.search(r"(reference|citation|source)", text, re.IGNORECASE):
        score += 0.5

    # 2. Length and depth (+2.0)
    word_count = len(text.split())
    if word_count > 150:
        score += 1.0
    if word_count > 300:
        score += 1.0

    # 3. Citation grounding (+1.5)
    if len(state.sources) > 0 and (
        re.search(r"\[.+\]", text) or "http" in text or "Source" in text
    ):
        score += 1.5

    # 4. Multi-perspective analysis (+0.5)
    if state.analysis_notes:
        score += 0.5

    return min(10.0, round(score, 1))


def calculate_citation_coverage(state: ResearchState) -> float:
    """Estimate citation coverage based on referenced sources."""
    if not state.final_answer or not state.sources:
        return 0.0

    answer_lower = state.final_answer.lower()
    has_generic_citations = bool(re.search(r"\[(source|doc|1|2|3|\d+).*?\]", answer_lower))
    cited_count = 0

    for doc in state.sources:
        source_id = str(doc.metadata.get("source_id", "")).lower()
        title_clean = re.sub(r"\[.*?\]", "", doc.title).strip().lower()
        title_keywords = [w for w in title_clean.split() if len(w) > 4]

        is_cited = (
            (source_id and source_id in answer_lower)
            or (title_clean and title_clean in answer_lower)
            or (title_keywords and any(k in answer_lower for k in title_keywords[:2]))
            or has_generic_citations
        )
        if is_cited:
            cited_count += 1

    return min(1.0, round(cited_count / max(1, len(state.sources)), 2))


def calculate_total_cost(state: ResearchState) -> float:
    """Sum cost from trace events and agent results."""
    total_cost = 0.0
    for event in state.trace:
        payload = event.get("payload", {})
        if isinstance(payload, dict) and "cost_usd" in payload:
            cost = payload.get("cost_usd")
            if cost is not None:
                total_cost += float(cost)

    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost is not None:
            total_cost += float(cost)

    return round(total_cost, 6)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and failure rate."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    has_failed = bool(state.errors or not state.final_answer)
    quality = evaluate_quality(state)
    citation_cov = calculate_citation_coverage(state)
    cost = calculate_total_cost(state)
    if cost == 0.0 and state.final_answer:
        cost = round(len(state.final_answer) / 4 * (0.6 / 1_000_000), 6)

    notes = f"Routes: {' -> '.join(state.route_history)}" if state.route_history else "Single-step"
    if has_failed:
        notes += f" | Errors: {len(state.errors)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=citation_cov,
        failure_rate=1.0 if has_failed else 0.0,
        notes=notes,
    )
    return state, metrics
