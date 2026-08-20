"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_external_tracing
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_external_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run single-agent baseline research."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(
        Panel(f"[bold cyan]Query:[/bold cyan] {request.query}", title="Single-Agent Baseline")
    )

    started = perf_counter()
    search_client = SearchClient()
    llm = LLMClient()

    # 1. Direct search
    sources = search_client.search(request.query, max_results=request.max_sources)
    state.sources = sources

    # 2. Direct synthesis in one prompt
    sources_text = "\n\n".join(
        f"[{idx + 1}] Title: {doc.title}\nURL: {doc.url or 'N/A'}\nSnippet: {doc.snippet}"
        for idx, doc in enumerate(sources)
    )
    system_prompt = (
        "You are an AI research assistant. Provide a concise, factual summary addressing "
        "the user query based directly on the provided search results with inline citations."
    )
    user_prompt = (
        f"Query: {request.query}\n\nSearch Results:\n{sources_text}\n\n"
        "Provide your research summary:"
    )

    resp = llm.complete(system_prompt, user_prompt)
    state.final_answer = resp.content
    state.record_route("single_agent")
    state.add_agent_result(
        agent=AgentName.WRITER,
        content=resp.content,
        metadata={"cost_usd": resp.cost_usd, "tokens": resp.output_tokens},
    )
    latency = perf_counter() - started

    console.print(
        Panel(state.final_answer, title="[bold green]Baseline Response[/bold green]", expand=False)
    )

    # Print summary metrics
    table = Table(title="Baseline Execution Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latency", f"{latency:.2f}s")
    table.add_row("Sources Retrieved", str(len(state.sources)))
    table.add_row("Input Tokens", str(resp.input_tokens or "N/A"))
    table.add_row("Output Tokens", str(resp.output_tokens or "N/A"))
    table.add_row("Estimated Cost", f"${resp.cost_usd:.5f}" if resp.cost_usd else "N/A")
    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor -> Researcher -> Analyst -> Writer -> Critic)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(
        Panel(
            f"[bold magenta]Query:[/bold magenta] {request.query}",
            title="Multi-Agent Research Workflow",
        )
    )

    workflow = MultiAgentWorkflow()
    started = perf_counter()
    result = workflow.run(state)
    latency = perf_counter() - started

    # Display Route Progression
    route_str = " ➔ ".join(f"[bold yellow]{r}[/bold yellow]" for r in result.route_history)
    console.print(Panel(route_str, title="[bold blue]Route Progression[/bold blue]"))

    # Display Final Answer
    if result.final_answer:
        console.print(
            Panel(
                result.final_answer,
                title="[bold green]Final Research Report[/bold green]",
                expand=False,
            )
        )

    # Display Critic Review if present
    if result.critic_notes:
        console.print(
            Panel(
                result.critic_notes,
                title="[bold cyan]Critic Verification[/bold cyan]",
                expand=False,
            )
        )

    # Display Metrics Summary
    table = Table(title="Multi-Agent Execution Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Latency", f"{latency:.2f}s")
    table.add_row("Iterations / Handoffs", str(result.iteration))
    table.add_row("Sources Gathered", str(len(result.sources)))
    table.add_row("Pipeline Stages", str(len(result.agent_results)))
    table.add_row("Trace Spans Recorded", str(len(result.trace)))
    console.print(table)


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art"
    ),
) -> None:
    """Run side-by-side benchmark comparing baseline and multi-agent systems."""
    _init()
    console.print(
        Panel(f"[bold yellow]Benchmarking query:[/bold yellow] {query}", title="Benchmark Runner")
    )

    def run_baseline_fn(q: str) -> ResearchState:
        s = ResearchState(request=_parse_query(q))
        docs = SearchClient().search(q, max_results=s.request.max_sources)
        s.sources = docs
        resp = LLMClient().complete(
            "Research assistant baseline", f"Query: {q}\nSources: {len(docs)}"
        )
        s.final_answer = resp.content
        s.record_route("baseline")
        s.add_agent_result(AgentName.WRITER, resp.content, {"cost_usd": resp.cost_usd})
        return s

    def run_multi_fn(q: str) -> ResearchState:
        s = ResearchState(request=_parse_query(q))
        return MultiAgentWorkflow().run(s)

    console.print("[dim]Running Single-Agent Baseline...[/dim]")
    _, baseline_m = run_benchmark("Single-Agent Baseline", query, run_baseline_fn)

    console.print("[dim]Running Multi-Agent Workflow...[/dim]")
    _, multi_m = run_benchmark("Multi-Agent Workflow", query, run_multi_fn)

    report = render_markdown_report([baseline_m, multi_m])
    console.print(Panel(report, title="[bold green]Benchmark Markdown Report[/bold green]"))


if __name__ == "__main__":
    app()
