"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily and offline corpus fallback."""

    def __init__(self, api_key: str | None = None, corpus_dir: Path | str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if corpus_dir:
            self.corpus_dir = Path(corpus_dir)
        else:
            # Look for ai_agent_offline_research_corpus_v2 in project root
            self.corpus_dir = (
                Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2"
            )
        self._corpus_cache: list[dict[str, Any]] | None = None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._is_valid_api_key(self.api_key):
            try:
                results = self._tavily_search(query, max_results)
                if results:
                    return results
            except Exception as exc:
                logger.warning("Tavily search failed (%s). Falling back to offline corpus.", exc)

        return self._offline_corpus_search(query, max_results)

    @staticmethod
    def _is_valid_api_key(key: str | None) -> bool:
        if not key or key.strip() in {"", "...", "tvly-...", "your_tavily_api_key"}:
            return False
        return len(key.strip()) > 10

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        import requests

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results: list[SourceDocument] = []
        for item in data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Search Result"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        return results

    def _load_corpus(self) -> list[dict[str, Any]]:
        if self._corpus_cache is not None:
            return self._corpus_cache

        docs: list[dict[str, Any]] = []
        topics_dir = self.corpus_dir / "topics"
        if topics_dir.exists() and topics_dir.is_dir():
            for json_path in topics_dir.glob("*.json"):
                try:
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                    topic_title = data.get("topic", json_path.stem)
                    kb = data.get("knowledge_base", {})

                    # Extract knowledge articles
                    for art in kb.get("knowledge_articles", []):
                        art_id = art.get("article_id", "art")
                        snippet_text = art.get("summary") or (art.get("content", "")[:400] + "...")
                        docs.append(
                            {
                                "title": f"[{topic_title}] {art.get('title', 'Article')}",
                                "url": f"corpus://{json_path.stem}/{art_id}",
                                "snippet": snippet_text,
                                "metadata": {
                                    "article_id": art_id,
                                    "topic": topic_title,
                                    "is_synthetic": art.get("is_synthetic", False),
                                },
                            }
                        )

                    # Extract source documents
                    for src in kb.get("source_documents", []):
                        src_id = src.get("source_id", "src")
                        docs.append(
                            {
                                "title": f"[{topic_title}] {src.get('title', 'Document')}",
                                "url": src.get("url") or f"corpus://{json_path.stem}/{src_id}",
                                "snippet": src.get("abstract") or src.get("snippet", ""),
                                "metadata": {
                                    "source_id": src_id,
                                    "topic": topic_title,
                                    "is_synthetic": src.get("is_synthetic", False),
                                },
                            }
                        )

                    # Extract key facts
                    for fact in kb.get("fact_bank", []):
                        fact_id = fact.get("fact_id", "fact")
                        claim = fact.get("claim", "")
                        ev = fact.get("evidence_snippet", "")
                        docs.append(
                            {
                                "title": f"[{topic_title}] Fact: {claim[:60]}",
                                "url": f"corpus://{json_path.stem}/{fact_id}",
                                "snippet": f"{claim} (Evidence: {ev})",
                                "metadata": {
                                    "fact_id": fact_id,
                                    "topic": topic_title,
                                    "confidence": fact.get("confidence"),
                                },
                            }
                        )
                except Exception as exc:
                    logger.debug("Failed reading corpus topic %s: %s", json_path.name, exc)

        self._corpus_cache = docs
        return self._corpus_cache

    def _offline_corpus_search(self, query: str, max_results: int) -> list[SourceDocument]:
        all_docs = self._load_corpus()
        if not all_docs:
            # Fallback default documents if corpus directory not loaded
            return [
                SourceDocument(
                    title="Role Specialization in Multi-Agent Systems",
                    url="https://arxiv.org/abs/2308.08155",
                    snippet=(
                        "Specialization of roles (Supervisor, Researcher, Analyst, Writer) "
                        "reduces individual agent context cognitive load and enables verification."
                    ),
                    metadata={"source_id": "SRC-MAS-01", "is_synthetic": False},
                ),
                SourceDocument(
                    title="Shared State and Context Engineering in Multi-Agent Workflows",
                    url="https://langchain-ai.github.io/langgraph/",
                    snippet=(
                        "Typed state schemas provide deterministic handoff boundaries "
                        "and prevent cascading hallucination in cyclic LLM graphs."
                    ),
                    metadata={"source_id": "SRC-MAS-02", "is_synthetic": False},
                ),
                SourceDocument(
                    title="Production Guardrails and Failure Modes for Autonomous AI Agents",
                    url="https://www.anthropic.com/engineering/building-effective-agents",
                    snippet=(
                        "Long-horizon tasks require max iterations, timeouts, fallback "
                        "handlers, and citation validation to avoid infinite loops."
                    ),
                    metadata={"source_id": "SRC-MAS-03", "is_synthetic": False},
                ),
            ][:max_results]

        tokens = set(re.findall(r"\w+", query.lower()))
        scored_docs: list[tuple[float, dict[str, Any]]] = []

        for doc in all_docs:
            text = f"{doc['title']} {doc['snippet']}".lower()
            doc_tokens = set(re.findall(r"\w+", text))
            intersection = tokens.intersection(doc_tokens)
            if intersection:
                score = sum(1.0 + (len(t) * 0.1) for t in intersection)
                if any(t in doc["title"].lower() for t in tokens):
                    score *= 1.5
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results: list[SourceDocument] = []
        for score, doc in scored_docs[:max_results]:
            meta = dict(doc.get("metadata", {}))
            meta["score"] = round(score, 2)
            results.append(
                SourceDocument(
                    title=doc["title"],
                    url=doc["url"],
                    snippet=doc["snippet"],
                    metadata=meta,
                )
            )

        if not results:
            # Fallback to first N docs
            for doc in all_docs[:max_results]:
                results.append(
                    SourceDocument(
                        title=doc["title"],
                        url=doc["url"],
                        snippet=doc["snippet"],
                        metadata=doc.get("metadata", {}),
                    )
                )

        return results
