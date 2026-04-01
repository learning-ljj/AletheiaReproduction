"""Portable agent orchestration runtime.

This module captures the architectural core of the source repository:
1. the main agent decides whether to delegate,
2. the runtime calls a subagent,
3. the subagent returns a structured report,
4. the main agent continues with that report in state.

Unlike the source repository, this version does not require LangGraph.

中文说明：
该模块实现了一个轻量级的运行时编排器（Runtime），负责将主策略的决策
（委派或返回最终答案）与注册的子 agent（subagents）对接。流程为：
主 agent 决策 -> 运行时执行子 agent -> 子 agent 返回结构化报告 -> 主 agent 继续决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .shared_types import (
    AgentTask,
    AgentWorker,
    DelegateAction,
    EventSink,
    FinalAction,
    MainAgentStrategy,
    RunEvent,
    SubAgentReport,
)


class InMemoryEventSink:
    def __init__(self) -> None:
        # Events are simply appended in order.
        # 以列表形式按序保存运行时事件，便于测试或回放。
        self.events: list[RunEvent] = []

    async def emit(self, event: RunEvent) -> None:
        # Store the event so callers can inspect the whole run later.
        # 将事件追加到内存列表中。
        self.events.append(event)


@dataclass(slots=True)
class RuntimeState:
    # Original user goal or request.
    # 用户最初的目标或请求。
    user_goal: str
    # 报告按子 agent 名称索引，便于快速查找和合并。
    reports: dict[str, SubAgentReport] = field(default_factory=dict)
    # 记录已委派的任务列表，用于可观测性与审计。
    tasks: list[AgentTask] = field(default_factory=list)
    # 当主循环结束并返回最终答案时填充该字段。
    final_answer: str = ""


class AgentRuntime:
    """Coordinate the main agent and a set of subagents."""
    # AgentRuntime 负责：
    # - 按轮次驱动主策略（MainAgentStrategy）决定下一步动作；
    # - 在需要时委派任务给注册的子 agent 并收集其结构化报告；
    # - 通过事件槽（EventSink）异步发出运行时事件，供 UI/日志/测试使用。

    def __init__(
        self,
        main_strategy: MainAgentStrategy,
        subagents: dict[str, AgentWorker],
        event_sink: EventSink | None = None,
        max_rounds: int = 8,
    ) -> None:
        # The strategy object owns "what to do next".
        # 主策略：决定接下来是委派任务还是返回最终答案。
        self._main_strategy = main_strategy
        # 按名称注册的子 agent，运行时动态查找并调用。
        self._subagents = subagents
        # 事件槽用于解耦运行时与具体的日志/监控/前端。
        self._event_sink = event_sink or InMemoryEventSink()
        # 最大轮数限制，防止无限委派循环。
        self._max_rounds = max_rounds

    async def run(self, user_goal: str) -> FinalAction:
        """Run the orchestration loop until the main agent returns a final answer."""
        # 为此次请求创建运行时状态容器。
        state = RuntimeState(user_goal=user_goal)

        # 发出启动事件，以便日志或 UI 初始化视图。
        await self._emit(
            "runtime_started",
            {
                "user_goal": user_goal,
            },
        )

        # 有界循环：以轮次为单位查询主策略的下一步决策。
        for round_index in range(1, self._max_rounds + 1):
            # 询问主策略下一步动作（可能是委派或返回最终答案）。
            action = await self._main_strategy.next_action(
                user_goal=state.user_goal,
                reports=state.reports,
            )

            # 如果是委派动作，则运行对应子 agent。
            if isinstance(action, DelegateAction):
                # 在执行前记录任务，以便可观测性（回放 / 审计）。
                state.tasks.append(action.task)
                # 发出委派事件，包含轮次与任务信息。
                await self._emit(
                    "delegate",
                    {
                        "round": round_index,
                        "task_id": action.task.task_id,
                        "target_agent": action.task.target_agent,
                        "goal": action.task.goal,
                        "payload": action.task.payload,
                    },
                )

                # 执行子 agent 并获取结构化报告。
                report = await self._run_subagent(action.task)

                # 将子 agent 的报告合并入运行状态，供主策略下一轮使用。
                state.reports[report.source_agent] = report

                # 发出子 agent 完成事件，便于外部系统展示或断言。
                await self._emit(
                    "subagent_completed",
                    {
                        "round": round_index,
                        "task_id": report.task_id,
                        "source_agent": report.source_agent,
                        "summary": report.summary,
                        "structured_output": report.structured_output,
                        "artifacts": report.artifacts,
                    },
                )
                # 继续下一轮，让主策略消费新报告。
                continue

            # 如果收到 FinalAction，说明主策略决定终止并返回答案。
            if isinstance(action, FinalAction):
                # 将答案保存在状态中以备检查。
                state.final_answer = action.answer
                # 发出最终答案事件后返回结果。
                await self._emit(
                    "final_answer",
                    {
                        "round": round_index,
                        "answer": action.answer,
                        "artifacts": action.artifacts,
                        "citations": [
                            {"title": item.title, "url": item.url}
                            for item in action.citations
                        ],
                    },
                )
                return action

            # 未知动作类型视为编程错误并抛出。
            raise TypeError(f"Unsupported action type: {type(action)!r}")

        # 超出最大轮次仍未产生最终答案，抛出异常以提示调用方。
        raise RuntimeError(
            "AgentRuntime hit max_rounds before the main agent produced a final answer"
        )

    async def _run_subagent(self, task: AgentTask) -> SubAgentReport:
        """Execute one delegated task and return the subagent report."""

        # 动态查找目标子 agent，保持运行时的可扩展性。
        worker = self._subagents.get(task.target_agent)

        # 如果未找到目标子 agent，尽早以清晰错误失败。
        if worker is None:
            raise KeyError(f"Unknown subagent: {task.target_agent}")

        # 在子 agent 执行前发出启动事件。
        await self._emit(
            "subagent_started",
            {
                "task_id": task.task_id,
                "target_agent": task.target_agent,
                "goal": task.goal,
            },
        )

        # 执行子 agent 并返回其结构化报告。
        return await worker.run(task)

    async def _emit(self, event_type: str, payload: dict) -> None:
        """Emit one runtime event through the configured sink."""
        # 将事件封装为 RunEvent 并通过事件槽发送。
        await self._event_sink.emit(
            RunEvent(
                event_type=event_type,
                payload=payload,
            )
        )
