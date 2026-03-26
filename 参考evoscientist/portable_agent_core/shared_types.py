"""Shared types for the low-coupling portable agent core.

The goal of this module is to keep every other module free from framework
dependencies such as LangChain, DeepAgents, or a specific model SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class SearchHit:
    # The human-readable title shown to the main agent or the user.
    title: str
    # The canonical URL that will later be fetched for deeper extraction.
    url: str
    # A short snippet returned by the search provider, if available.
    snippet: str = ""
    # Optional numeric score exposed by the provider.
    score: float = 0.0
    # Provider name helps debugging and later ranking decisions.
    provider: str = ""
    # Free-form metadata keeps the transport layer flexible.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    # The title is what the main agent will show in the final answer.
    title: str
    # The URL is the pointer to the original source.
    url: str
    # Provider helps explain where the citation came from.
    provider: str = ""


@dataclass(slots=True)
class ExtractedDocument:
    # Source URL or local path.
    source: str
    # MIME type guides downstream parsing choices.
    mime_type: str
    # Title can be derived from HTML or PDF metadata.
    title: str = ""
    # Markdown keeps formatting while staying model-friendly.
    markdown: str = ""
    # Plain text is useful for embeddings or lightweight validation.
    text: str = ""
    # Metadata stays open-ended on purpose.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTask:
    # A stable task id makes logs and retries easier to follow.
    task_id: str
    # The target subagent name, such as "literature_search".
    target_agent: str
    # Human-readable goal for the subagent.
    goal: str
    # Structured payload keeps coupling low between agents.
    payload: dict[str, Any] = field(default_factory=dict)
    # Context lets the main agent pass prior findings forward.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubAgentReport:
    # Echo the originating task id for correlation.
    task_id: str
    # Identify which subagent produced the report.
    source_agent: str
    # Summary is the compressed handoff text consumed by the main agent.
    summary: str
    # Structured data allows deterministic downstream use.
    structured_output: dict[str, Any] = field(default_factory=dict)
    # Citations flow back into the main answer builder.
    citations: list[Citation] = field(default_factory=list)
    # Artifacts can store generated file paths or URLs.
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DelegateAction:
    # The task that should be executed by a subagent.
    task: AgentTask


@dataclass(slots=True)
class FinalAction:
    # Final natural-language answer produced by the main agent.
    answer: str
    # Optional citations to attach to the answer.
    citations: list[Citation] = field(default_factory=list)
    # Optional artifacts generated during the run.
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunEvent:
    # Event type examples: "delegate", "subagent_completed", "final_answer".
    event_type: str
    # Payload shape is event-specific.
    payload: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    async def emit(self, event: RunEvent) -> None:
        """Consume runtime events emitted by the agent runtime."""


class AgentWorker(Protocol):
    name: str

    async def run(self, task: AgentTask) -> SubAgentReport:
        """Execute a delegated task and return a structured report."""


class MainAgentStrategy(Protocol):
    async def next_action(
        self,
        user_goal: str,
        reports: dict[str, SubAgentReport],
    ) -> DelegateAction | FinalAction:
        """Choose between delegating work and returning a final answer."""


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Return normalized search hits for a literature-style query."""
