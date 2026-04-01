"""SearcherAgent retrieval chain implementation."""

from __future__ import annotations

from typing import Callable

from src.memory.problem_memory import ProblemMemory
from src.tools.search import (
    build_paper_filename,
    build_paper_markdown,
    dedup_papers,
    expand_queries,
    multi_source_search,
)


class SearcherAgent:
    """Stable retrieval chain: expand -> multi-source -> dedup -> summarize -> persist."""

    def __init__(
        self,
        *,
        problem_memory: ProblemMemory,
        source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None,
        limit_per_query: int = 10,
    ):
        self.problem_memory = problem_memory
        self.source_handlers = source_handlers or {}
        self.limit_per_query = limit_per_query

    def run(self, *, query: str | None = None, query_bundle: list[str] | None = None) -> dict:
        expanded_queries = expand_queries(query=query, query_bundle=query_bundle)
        raw_hits = multi_source_search(
            expanded_queries,
            source_handlers=self.source_handlers,
            limit_per_query=self.limit_per_query,
        )
        unique_hits = dedup_papers(raw_hits)

        papers: list[dict] = []
        for paper in unique_hits:
            markdown = build_paper_markdown(paper)
            filename = build_paper_filename(paper)
            path = self.problem_memory.add_paper(markdown, filename=filename)
            papers.append(
                {
                    "path": str(path),
                    "layer1": {
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "arxiv_id": paper.get("arxiv_id"),
                    },
                    "source": paper.get("source"),
                }
            )

        return {
            "queries": expanded_queries,
            "papers": papers,
            "count": len(papers),
        }
