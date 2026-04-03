"""Worklog summary agent built on top of BaseAgent runtime."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.llm_client import LLMResponse


class WorklogSummaryAgent(BaseAgent):
    """Single-shot JSON summary agent for offline worklog rendering."""

    def __init__(self, *, llm_client, system_prompt: str):
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=[],
            tool_executor=None,
            max_tool_rounds=1,
            stream_prefix="WORKLOG",
        )

    def run_summary(self, prompt_text: str) -> LLMResponse:
        return super().run(prompt_text)
