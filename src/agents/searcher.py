"""Searcher 子代理实现。

核心职责（按执行顺序）：
1) 扩展查询；
2) 调用多来源检索；
3) 去重；
4) 转 markdown 并落盘；
5) 返回给主链一个“可追踪 + 可恢复”的结构化结果。
"""

from __future__ import annotations

from typing import Callable

from src.memory.problem_memory import ProblemMemory
from src.tools.search_papers.search import (
    build_paper_filename,
    build_paper_markdown,
    dedup_papers,
    expand_queries,
    multi_source_search_with_diagnostics,
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
        # 第一步：把 query 变成“检索更友好”的 query 列表。
        expanded_queries = expand_queries(query=query, query_bundle=query_bundle)

        # 第二步：执行多来源检索。
        # 这里拿到的不只是 results，还包含 errors/recovered_errors。
        # 这样就算某个来源挂了，也能把错误上下文交给 LLM，而不是静默丢失。
        search_diagnostics = multi_source_search_with_diagnostics(
            expanded_queries,
            source_handlers=self.source_handlers,
            limit_per_query=self.limit_per_query,
        )
        raw_hits = search_diagnostics.get("results", [])
        search_errors = search_diagnostics.get("errors", [])
        recovered_errors = search_diagnostics.get("recovered_errors", [])
        stats = search_diagnostics.get("stats", {})

        # 第三步：按 DOI/arXiv/title 进行去重，避免重复文献污染上下文。
        unique_hits = dedup_papers(raw_hits)

        papers: list[dict] = []
        for paper in unique_hits:
            # 第四步：将单条论文写成三层 markdown，并落到 artifact/papers。
            markdown = build_paper_markdown(paper)
            filename = build_paper_filename(paper)
            path = self.problem_memory.add_paper(markdown, filename=filename)

            candidate_claims: list[str] = []
            for line in markdown.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    candidate_claims.append(stripped[2:].strip())

            papers.append(
                {
                    "path": str(path),
                    "layer1": {
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "arxiv_id": paper.get("arxiv_id"),
                    },
                    "source": paper.get("source"),
                    "source_list": paper.get("source_list", []),
                    "candidate_claims": [item for item in candidate_claims if item and item.upper() != "NONE"],
                }
            )

        # 第五步：把“成功结果 + 失败细节”一并返回。
        # 上层 LLM 可以利用这些失败信息调整下一轮 query、换来源、或切换证明策略。
        return {
            "stages": {
                "query_expanded": len(expanded_queries),
                "raw_hits": len(raw_hits),
                "dedup_hits": len(unique_hits),
                "persisted_papers": len(papers),
                "attempts": int(stats.get("attempts", 0) or 0),
                "failed_queries": int(stats.get("failed_queries", 0) or 0),
                "recovered_queries": int(stats.get("recovered_queries", 0) or 0),
            },
            "queries": expanded_queries,
            "papers": papers,
            "count": len(papers),
            "errors": search_errors,
            "recovered_errors": recovered_errors,
            "error_count": len(search_errors),
            "has_errors": bool(search_errors),
            "llm_action_hint": (
                "检索链路存在来源级报错，请优先阅读 errors 字段后再决定：改写查询、缩小范围、"
                "或先基于已检索结果继续推理。"
                if search_errors
                else "检索链路无报错，可直接使用 papers 进行后续推理。"
            ),
        }
