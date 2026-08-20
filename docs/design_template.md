# Design Template: Multi-Agent Research System

> **Học viên:** Lê Văn Tuấn  
> **MSSV:** 2A202601016  
> **Bài Lab:** Lab 20 - Multi-Agent Research System

## Problem

Building an autonomous, evidence-grounded research assistant capable of accepting complex, open-ended research queries (e.g., *"Research GraphRAG state-of-the-art and production guardrails"*), retrieving relevant source documents, extracting factual claims, analyzing trade-offs, and synthesizing a comprehensive, well-structured technical report with verified inline citations and quality verification.

## Why multi-agent?

In complex technical research tasks, single-agent monolithic prompts suffer from several major bottlenecks:
1. **Context Congestion & Cognitive Overload**: Combining search query formulation, raw document retrieval, critical synthesis, contradiction resolution, and long-form prose generation into a single prompt frequently leads to hallucinations and missed claims.
2. **Lack of Critical Separation**: A single agent rarely critiques its own assertions with high objectivity.
3. **Role Specialization**: Decomposing the system into specialized nodes (Supervisor, Researcher, Analyst, Writer, Critic) enforces modular separation of concerns, explicit state handoffs, verifiable citation grounding, and automated quality gates.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Orchestrates workflow execution, evaluates pipeline state, and selects next worker | `ResearchState` | Next route decision (`state.route_history`) | Routing loops, infinite iteration cycling |
| **Researcher** | Retrieves relevant sources from search/corpus and extracts factual notes | `request.query`, `sources` | `state.sources`, `state.research_notes` | Retrieval irrelevance, token explosion from unranked raw documents |
| **Analyst** | Critically evaluates findings, identifies trade-offs, and checks claim evidence strength | `state.research_notes`, `state.sources` | `state.analysis_notes` | Biased interpretations, over-generalization without factual backing |
| **Writer** | Synthesizes technical report with proper Markdown structure and inline citations | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | Citation hallucination, omitting required analytical sections |
| **Critic** *(Optional/Verifier)* | Validates factuality, citation coverage, and completeness against user query | `state.final_answer`, `state.sources` | `state.critic_notes` | False negative rejections or unhelpful shallow critiques |

## Shared state

The `ResearchState` Pydantic model serves as the single source of truth across all graph transitions:

- `request` (`ResearchQuery`): Original user query, target audience, and retrieval limits.
- `iteration` (`int`): Tracks total handoff steps to enforce `max_iterations` guardrails.
- `route_history` (`list[str]`): Sequential record of all routing decisions for debugging and auditing.
- `sources` (`list[SourceDocument]`): Deduplicated list of discovered documents with metadata and snippet contents.
- `research_notes` (`str | None`): Raw factual extractions produced by the Researcher.
- `analysis_notes` (`str | None`): Synthesized trade-offs, failure modes, and analytical insights from the Analyst.
- `final_answer` (`str | None`): Formatted Markdown technical report synthesized by the Writer.
- `critic_notes` (`str | None`): Factuality, citation coverage, and quality evaluation from the Critic.
- `agent_results` (`list[AgentResult]`): Audit log of each agent's contribution and token/cost metadata.
- `trace` (`list[dict[str, Any]]`): Fine-grained span telemetry capturing execution time and token metrics.
- `errors` (`list[str]`): List of runtime exceptions or warnings captured during workflow execution.

## Routing policy

```text
               +---------------------------+
               |        User Query         |
               +-------------+-------------+
                             |
                             v
               +-------------+-------------+
       +------>|     Supervisor / Router   |<------+
       |       +-------------+-------------+       |
       |                     |                     |
       |  [need sources]     | [has sources]       |
       |                     v                     |
+------+------+       +------+------+       +------+------+
| Researcher  |       |   Analyst   |       |   Writer    |
+------+------+       +------+------+       +------+------+
       |                     |                     |
       +---------------------+---------------------+
                             |
                   [has final answer]
                             v
                      +------+------+
                      |   Critic    |
                      +------+------+
                             |
                             v
                         [ Done ]
```

## Guardrails

- **Max iterations**: Enforced limit (`max_iterations = 6`) to prevent infinite looping and runaway token costs.
- **Timeout**: Per-step and total workflow timeout (`timeout_seconds = 60s`).
- **Retry**: Exponential backoff retry with `tenacity` on LLM network/rate-limit errors.
- **Fallback**: Graceful fallback to offline research corpus (`ai_agent_offline_research_corpus_v2`) and local heuristic generator if API keys or network are unavailable.
- **Validation**: Schema-level Pydantic validation on all inputs, query lengths, states, and agent handoffs.

## Benchmark plan

- **Queries Tested**:
  1. *"Research GraphRAG state-of-the-art and write a 500-word summary"*
  2. *"Compare single-agent and multi-agent workflows for customer support"*
  3. *"Summarize production guardrails for LLM agents"*
- **Target Metrics**: Latency (s), Cost (USD), Quality Score (0-10), Citation Coverage (0-100%), Failure Rate (%).
- **Expected Outcome**: Multi-Agent system delivers higher quality (+3.0 to +4.0 points) and superior citation coverage (+40%) at the expense of higher latency (2-4x) and token cost.
