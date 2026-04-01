"""Minimal runnable demo that wires the portable modules together.

Run with:
    python -m migration_examples.portable_agent_core.demo_research_runtime

This demo uses a static search provider so it can run without network access.
Swap StaticSearchProvider for TavilySearchProvider in real usage.

中文说明：
这是一个最小可运行示例，展示如何组装 `agent_framework`、检索与抽取子 agent，
以及主策略（ResearchCoordinatorStrategy）如何按轮次委派并合并子 agent 的报告。
示例使用静态搜索结果以便离线运行和复现。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import count
from typing import Any

from .agent_framework import AgentRuntime, InMemoryEventSink
from .content_extraction import UniversalContentExtractor
from .literature_search import (
    MultiProviderLiteratureSearch,
    SearchBundle,
    StaticSearchProvider,
    citations_from_bundle,
)
from .shared_types import (
    AgentTask,
    AgentWorker,
    DelegateAction,
    FinalAction,
    SearchHit,
    SubAgentReport,
)


# Task ids are generated from a simple counter to keep the demo deterministic.
_TASK_COUNTER = count(1)


def new_task_id() -> str:
    # Return ids such as "task-1", "task-2", and so on.
    return f"task-{next(_TASK_COUNTER)}"
    # new_task_id 用于生成稳定且可追踪的任务 id，便于在事件流中定位任务。


class LiteratureSearchSubAgent:
    """Subagent responsible for the search phase."""

    name = "literature_search"

    def __init__(self, search_service: MultiProviderLiteratureSearch) -> None:
        # The subagent depends only on the normalized search service.
        self._search_service = search_service

    async def run(self, task: AgentTask) -> SubAgentReport:
        # Read the query from the delegated payload.
        query = str(task.payload["query"])
        # Execute the normalized search service.
        bundle = await self._search_service.search(query=query, max_results=5)
        # Build a short summary for the main agent handoff.
        summary = build_search_summary(bundle)
        # Return a structured report that the main agent can consume deterministically.
        return SubAgentReport(
            task_id=task.task_id,
            source_agent=self.name,
            summary=summary,
            structured_output={
                "query": bundle.query,
                "providers_used": bundle.providers_used,
                "hits": [
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "snippet": hit.snippet,
                    }
                    for hit in bundle.hits
                ],
            },
            citations=citations_from_bundle(bundle),
            artifacts=[],
        )
    # 该子 agent 的 run 方法演示了如何把服务返回的 SearchBundle 转成 SubAgentReport，
    # 包含可供主 agent 程序化使用的 structured_output 以及可供展示的 summary。


class ContentExtractionSubAgent:
    """Subagent responsible for the extraction phase."""

    name = "content_extraction"

    def __init__(self, extractor: UniversalContentExtractor) -> None:
        # The extractor is the only dependency this subagent needs.
        self._extractor = extractor

    async def run(self, task: AgentTask) -> SubAgentReport:
        # URLs arrive from the previous literature search report.
        urls = list(task.payload.get("urls", []))
        # Keep the demo bounded and cheap.
        selected_urls = urls[:2]

        # Extract every selected URL sequentially for clarity.
        extracted_documents = []
        for url in selected_urls:
            try:
                extracted_documents.append(await self._extractor.extract(url))
            except Exception as exc:  # noqa: BLE001
                extracted_documents.append(
                    {
                        "source": url,
                        "title": "",
                        "text": f"Extraction failed: {exc}",
                        "markdown": "",
                    }
                )

        # Normalize the extraction outputs into dictionaries.
        normalized_docs: list[dict[str, Any]] = []
        for item in extracted_documents:
            if isinstance(item, dict):
                normalized_docs.append(item)
                continue
            normalized_docs.append(
                {
                    "source": item.source,
                    "title": item.title,
                    "text": item.text[:1500],
                    "markdown": item.markdown[:1500],
                }
            )

        # Create a concise report summary for the main agent.
        summary = build_extraction_summary(normalized_docs)

        return SubAgentReport(
            task_id=task.task_id,
            source_agent=self.name,
            summary=summary,
            structured_output={
                "documents": normalized_docs,
            },
            citations=[],
            artifacts=[],
        )
    # 该子 agent 演示了从 URL 列表中抽取内容、做容错（异常捕获）并
    # 将结果归一化为字典列表，供主 agent 汇总使用。


@dataclass(slots=True)
class ResearchCoordinatorStrategy:
    """A small main-agent strategy used only for the migration demo."""

    async def next_action(
        self,
        user_goal: str,
        reports: dict[str, SubAgentReport],
    ) -> DelegateAction | FinalAction:
        # First round: if no search report exists, delegate literature search.
        if "literature_search" not in reports:
            return DelegateAction(
                task=AgentTask(
                    task_id=new_task_id(),
                    target_agent="literature_search",
                    goal="Find literature sources relevant to the user request",
                    payload={"query": user_goal},
                )
            )

        # Second round: if search exists but extraction does not, delegate extraction.
        if "content_extraction" not in reports:
            search_report = reports["literature_search"]
            urls = [
                item["url"]
                for item in search_report.structured_output.get("hits", [])
                if item.get("url")
            ]
            return DelegateAction(
                task=AgentTask(
                    task_id=new_task_id(),
                    target_agent="content_extraction",
                    goal="Extract readable content from the top literature URLs",
                    payload={"urls": urls},
                    context={"search_summary": search_report.summary},
                )
            )

        # Final round: merge both reports into a user-facing answer.
        return FinalAction(
            answer=build_final_answer(
                user_goal=user_goal,
                search_report=reports["literature_search"],
                extraction_report=reports["content_extraction"],
            ),
            citations=reports["literature_search"].citations,
        )
    # 这是一个非常小的主策略示例：按顺序完成两个阶段（检索 -> 抽取），
    # 并最终把子报告合并为最终答案。真实场景下主策略会更复杂。


def build_search_summary(bundle: SearchBundle) -> str:
    # Summarize the top hits in a compact, model-friendly sentence.
    lines = [f"Search query: {bundle.query}"]
    for hit in bundle.hits[:3]:
        lines.append(f"- {hit.title} | {hit.url}")
    return "\n".join(lines)
    # 该函数生成简短的检索摘要，供主 agent 在下一阶段或输出中使用。


def build_extraction_summary(documents: list[dict[str, Any]]) -> str:
    # Summarize only the first chunk of each extracted document.
    lines = ["Extracted document summaries:"]
    for doc in documents:
        preview = str(doc.get("text", "") or doc.get("markdown", ""))[:220]
        lines.append(f"- {doc.get('source', '')}: {preview}")
    return "\n".join(lines)
    # 将抽取到的文档做简要预览，便于主 agent 快速读取要点。


def build_final_answer(
    user_goal: str,
    search_report: SubAgentReport,
    extraction_report: SubAgentReport,
) -> str:
    # Merge the subagent outputs the same way a main agent would merge reports.
    lines = [f"User goal: {user_goal}", ""]
    lines.append("Search phase handoff:")
    lines.append(search_report.summary)
    lines.append("")
    lines.append("Extraction phase handoff:")
    lines.append(extraction_report.summary)
    lines.append("")
    lines.append("Main-agent interpretation:")
    lines.append(
        "The main agent first delegated discovery to the search subagent, then "
        "delegated URL parsing to the extraction subagent, and finally merged "
        "both structured reports into one answer."
    )
    return "\n".join(lines)
    # build_final_answer 把子 agent 的摘要直接拼接成最终可读回答，在真实系统中
    # 主 agent 可能会在此基础上进行更复杂的归纳或生成。


async def main() -> None:
    # Create deterministic static hits so the demo can run offline.
    static_hits = [
        SearchHit(
            title="GraphRAG Survey Paper",
            url="https://example.org/papers/graphrag-survey",
            snippet="A survey of graph-augmented retrieval methods.",
            score=0.92,
            provider="static",
        ),
        SearchHit(
            title="Multi-Agent Literature Review Benchmark",
            url="https://example.org/papers/multi-agent-review-benchmark",
            snippet="A benchmark for long-horizon literature review agents.",
            score=0.87,
            provider="static",
        ),
    ]

    # Build the low-coupling search service.
    search_service = MultiProviderLiteratureSearch(
        providers=[StaticSearchProvider(static_hits)],
    )

    # Build the low-coupling extraction service.
    extractor = UniversalContentExtractor()

    # Register subagents by name so the runtime can delegate dynamically.
    subagents: dict[str, AgentWorker] = {
        "literature_search": LiteratureSearchSubAgent(search_service),
        "content_extraction": ContentExtractionSubAgent(extractor),
    }

    # Store runtime events so the user can inspect the interaction trace.
    sink = InMemoryEventSink()

    # Build the runtime with a simple coordinator strategy.
    runtime = AgentRuntime(
        main_strategy=ResearchCoordinatorStrategy(),
        subagents=subagents,
        event_sink=sink,
    )

    # Run one end-to-end request through the orchestration loop.
    result = await runtime.run(
        user_goal="Find literature about multi-agent scientific discovery systems"
    )

    # Print the final answer first.
    print("=== FINAL ANSWER ===")
    print(result.answer)
    print()

    # Then print the runtime event trace so the user can study the handoffs.
    print("=== EVENT TRACE ===")
    for event in sink.events:
        print(f"{event.event_type}: {event.payload}")

    # Finally print citations to show how search results flow into the final answer.
    print()
    print("=== CITATIONS ===")
    for citation in result.citations:
        print(f"- {citation.title} | {citation.url}")


if __name__ == "__main__":
    asyncio.run(main())
