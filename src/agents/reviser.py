"""Reviser agent implemented on top of BaseAgent."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.llm_client import LLMResponse


class ReviserAgent(BaseAgent):
    """Stateful reviser stage runtime."""

    def __init__(
        self,
        *,
        llm_client,
        system_prompt: str,
        tools: list[dict] | None = None,
        tool_executor=None,
        max_tool_rounds: int = 5,
    ):
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            stream_prefix="REVISER",
        )

    @staticmethod
    def _build_input(problem_text: str, previous_solution: str, verification_report: str) -> str:
        return (
            problem_text.strip()
            + "\n\n---\nPrevious Solution:\n"
            + (previous_solution or "").strip()
            + "\n\n---\nVerification Report:\n"
            + (verification_report or "").strip()
        )

    @staticmethod
    def _has_contract(text: str) -> bool:
        return "<solution>" in text and "</solution>" in text

    def run(
        self,
        *,
        problem_text: str,
        previous_solution: str,
        verification_report: str,
    ) -> LLMResponse:
        payload = self._build_input(problem_text, previous_solution, verification_report)
        content = super().run(payload)
        if self._has_contract(content):
            return LLMResponse(content=content, reasoning_content="")

        retry_payload = (
            payload
            + "\n\nFORMAT REQUIRED:\n"
            + "<verdict>...</verdict>\n"
            + "<solution>...</solution>"
        )
        retry_content = super().run(retry_payload)
        return LLMResponse(content=retry_content, reasoning_content="")
