"""Portable agent orchestration runtime.

This module captures the architectural core of the source repository:
1. the main agent decides whether to delegate,
2. the runtime calls a subagent,
3. the subagent returns a structured report,
4. the main agent continues with that report in state.

Unlike the source repository, this version does not require LangGraph.
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
        self.events: list[RunEvent] = []

    async def emit(self, event: RunEvent) -> None:
        # Store the event so callers can inspect the whole run later.
        self.events.append(event)


@dataclass(slots=True)
class RuntimeState:
    # Original user goal or request.
    user_goal: str
    # Reports are keyed by subagent name for easy lookup.
    reports: dict[str, SubAgentReport] = field(default_factory=dict)
    # Tasks let the runtime keep a full delegation trace.
    tasks: list[AgentTask] = field(default_factory=list)
    # Final answer is filled only when the loop terminates.
    final_answer: str = ""


class AgentRuntime:
    """Coordinate the main agent and a set of subagents."""

    def __init__(
        self,
        main_strategy: MainAgentStrategy,
        subagents: dict[str, AgentWorker],
        event_sink: EventSink | None = None,
        max_rounds: int = 8,
    ) -> None:
        # The strategy object owns "what to do next".
        self._main_strategy = main_strategy
        # Subagents are registered by name so delegation stays dynamic.
        self._subagents = subagents
        # Event sink decouples the runtime from any specific UI or logger.
        self._event_sink = event_sink or InMemoryEventSink()
        # max_rounds prevents accidental infinite delegation loops.
        self._max_rounds = max_rounds

    async def run(self, user_goal: str) -> FinalAction:
        """Run the orchestration loop until the main agent returns a final answer."""

        # Create a new runtime state for this user request.
        state = RuntimeState(user_goal=user_goal)

        # Emit a start event so logs or UIs can initialize their timeline.
        await self._emit(
            "runtime_started",
            {
                "user_goal": user_goal,
            },
        )

        # The bounded loop mirrors a controlled agent recursion limit.
        for round_index in range(1, self._max_rounds + 1):
            # Ask the main strategy what the next action should be.
            action = await self._main_strategy.next_action(
                user_goal=state.user_goal,
                reports=state.reports,
            )

            # DelegateAction means the runtime should call a subagent now.
            if isinstance(action, DelegateAction):
                # Record the task before execution for observability.
                state.tasks.append(action.task)
                # Emit a delegate event so the caller sees the handoff.
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

                # Execute the subagent and get a structured report back.
                report = await self._run_subagent(action.task)

                # Merge the report into state so the main agent can use it next round.
                state.reports[report.source_agent] = report

                # Emit the completed report for logs, UIs, or tests.
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
                # Continue the main loop with the newly merged report.
                continue

            # FinalAction means the main strategy is ready to stop.
            if isinstance(action, FinalAction):
                # Store the answer on state for completeness.
                state.final_answer = action.answer
                # Emit one final event before returning.
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

            # Any unknown action type is a programmer error.
            raise TypeError(f"Unsupported action type: {type(action)!r}")

        # If the loop finishes without a final answer, fail loudly.
        raise RuntimeError(
            "AgentRuntime hit max_rounds before the main agent produced a final answer"
        )

    async def _run_subagent(self, task: AgentTask) -> SubAgentReport:
        """Execute one delegated task and return the subagent report."""

        # Look up the target subagent dynamically so the runtime stays extensible.
        worker = self._subagents.get(task.target_agent)

        # Missing subagents should fail early with a clear message.
        if worker is None:
            raise KeyError(f"Unknown subagent: {task.target_agent}")

        # Emit a start event before the subagent does any work.
        await self._emit(
            "subagent_started",
            {
                "task_id": task.task_id,
                "target_agent": task.target_agent,
                "goal": task.goal,
            },
        )

        # Run the worker and return its structured report.
        return await worker.run(task)

    async def _emit(self, event_type: str, payload: dict) -> None:
        """Emit one runtime event through the configured sink."""

        await self._event_sink.emit(
            RunEvent(
                event_type=event_type,
                payload=payload,
            )
        )
