"""Shared types for the low-coupling portable agent core.

The goal of this module is to keep every other module free from framework
dependencies such as LangChain, DeepAgents, or a specific model SDK.

中文说明：
该模块定义了代理框架中使用的共享数据结构（DTO）和协议接口。
目标是保持其他模块与特定框架或 SDK 解耦，便于移植与测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class SearchHit:
    # The human-readable title shown to the main agent or the user.
    # 人类可读的标题，展示给主 agent 或最终用户。
    title: str
    # The canonical URL that will later be fetched for deeper extraction.
    # 对应的规范化 URL，后续会用于抓取并做深度抽取。
    url: str
    # A short snippet returned by the search provider, if available.
    # 检索提供者返回的简短摘要片段（若可用）。
    snippet: str = ""
    # Optional numeric score exposed by the provider.
    # 可选的数值评分，用于排序或调试。
    score: float = 0.0
    # Provider name helps debugging and later ranking decisions.
    # 提供者名称，便于调试与基于源的排序决策。
    provider: str = ""
    # Free-form metadata keeps the transport layer flexible.
    # 自由格式的元数据，用于传递额外信息（如日期、id 等）。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    # The title is what the main agent will show in the final answer.
    # 标题：主 agent 在最终答案中展示的引用标题。
    title: str
    # The URL is the pointer to the original source.
    # URL：指向原始来源的链接。
    url: str
    # Provider helps explain where the citation came from.
    # Provider：说明该引用来自哪个检索或抓取提供者。
    provider: str = ""


@dataclass(slots=True)
class ExtractedDocument:
    # Source URL or local path.
    # 来源：文档的原始 URL 或本地路径。
    source: str
    # MIME type guides downstream parsing choices.
    # MIME 类型：用于指导后续解析器（例如 text/html 或 application/pdf）。
    mime_type: str
    # Title can be derived from HTML or PDF metadata.
    # 标题：可从 HTML 的 <title> 或 PDF 元数据中提取。
    title: str = ""
    # Markdown keeps formatting while staying model-friendly.
    # Markdown：将提取的内容转换为 Markdown 格式，保留简单格式化。
    markdown: str = ""
    # Plain text is useful for embeddings or lightweight validation.
    # 纯文本：用于构建向量表示或快速校验。
    text: str = ""
    # Metadata stays open-ended on purpose.
    # 元数据：为开放型字段，可包含页数、作者、抓取时间等。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTask:
    # A stable task id makes logs and retries easier to follow.
    # 任务 ID：用于日志跟踪与重试关联。
    task_id: str
    # The target subagent name, such as "literature_search".
    # 目标子 agent 名称，例如 "literature_search"，表示要委派给哪个子 agent。
    target_agent: str
    # Human-readable goal for the subagent.
    # 对子 agent 的人类可读目标描述，方便理解任务意图。
    goal: str
    # Structured payload keeps coupling low between agents.
    # 结构化载荷：向子 agent 传递所需参数或上下文数据，尽量保持通用结构。
    payload: dict[str, Any] = field(default_factory=dict)
    # Context lets the main agent pass prior findings forward.
    # 上下文：主 agent 可以通过此字段把之前的发现或状态传递给子 agent。
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubAgentReport:
    # Echo the originating task id for correlation.
    # 报告关联的任务 ID，便于将报告与原始任务对应。
    task_id: str
    # Identify which subagent produced the report.
    # 报告来源：哪个子 agent 生成了该报告。
    source_agent: str
    # Summary is the compressed handoff text consumed by the main agent.
    # 摘要：压缩后的文本交付物，供主 agent 快速消费决策。
    summary: str
    # Structured data allows deterministic downstream use.
    # 结构化输出：机器可解析的字段，便于后续自动化处理。
    structured_output: dict[str, Any] = field(default_factory=dict)
    # Citations flow back into the main answer builder.
    # 引用列表：子 agent 返回的引用，会被主 agent 插入最终答案中。
    citations: list[Citation] = field(default_factory=list)
    # Artifacts can store generated file paths or URLs.
    # 人工产物：可以存放生成的文件路径、临时 URL 等。
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DelegateAction:
    # The task that should be executed by a subagent.
    # 表示应由子 agent 执行的委派任务对象。
    task: AgentTask


@dataclass(slots=True)
class FinalAction:
    # Final natural-language answer produced by the main agent.
    # 最终自然语言答案，由主 agent 生成并返回给调用方或用户。
    answer: str
    # Optional citations to attach to the answer.
    # 可选引用：附在答案后的参考来源列表。
    citations: list[Citation] = field(default_factory=list)
    # Optional artifacts generated during the run.
    # 可选产物：运行过程中生成的文件或资源链接列表。
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunEvent:
    # Event type examples: "delegate", "subagent_completed", "final_answer".
    # 事件类型示例："delegate"（委派任务）、"subagent_completed"（子 agent 完成）、"final_answer"（最终答案）。
    event_type: str
    # Payload shape is event-specific.
    # 事件负载：根据事件类型不同而变化，通常为字典结构。
    payload: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    async def emit(self, event: RunEvent) -> None:
        """Consume runtime events emitted by the agent runtime."""
        # 将运行时事件发送到监控、日志或前端展示层的抽象接口。


class AgentWorker(Protocol):
    name: str

    async def run(self, task: AgentTask) -> SubAgentReport:
        """Execute a delegated task and return a structured report."""
        # 子 agent 的运行接口：接收一个 AgentTask，返回 SubAgentReport。


class MainAgentStrategy(Protocol):
    async def next_action(
        self,
        user_goal: str,
        reports: dict[str, SubAgentReport],
    ) -> DelegateAction | FinalAction:
        """Choose between delegating work and returning a final answer."""
        # 主策略接口：根据用户目标与已获得的子报告决定下一步是继续委派还是返回最终答案。


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Return normalized search hits for a literature-style query."""
        # 搜索提供者接口：执行查询并返回统一规范的 SearchHit 列表，便于上层消化处理。
