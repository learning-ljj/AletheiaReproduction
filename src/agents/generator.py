"""Generator agent implemented on top of BaseAgent."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.llm_client import LLMResponse


class GeneratorAgent(BaseAgent):
    """Stateful generator stage runtime."""

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
            stream_prefix="GENERATOR",
        )

    @staticmethod
    def _build_input(
        problem_text: str,
        *,
        layer1_summaries: list[str] | None = None,
        error_lessons: str | None = None,
    ) -> str:
        parts = [problem_text.strip()]
        parts.append(
            "\n\nTooling Hint:\n"
            "- If external references are needed, call tool `call_searcher` with a focused query.\n"
            "- After retrieval, cite artifact paths as [cite:path]."
        )
        if layer1_summaries:
            parts.append("\n\n---\nLayer1 Summaries:\n" + "\n".join(f"- {item}" for item in layer1_summaries))
        if error_lessons:
            parts.append("\n\n---\nError Lessons:\n" + error_lessons.strip())
        return "".join(parts)

    @staticmethod
    def _has_contract(text: str) -> bool:
        return "<solution>" in text and "</solution>" in text and "<verdict>" in text and "</verdict>" in text

    def run(
        self,
        *,
        problem_text: str,
        layer1_summaries: list[str] | None = None,
        error_lessons: str | None = None,
    ) -> LLMResponse:
        payload = self._build_input(
            problem_text,
            layer1_summaries=layer1_summaries,
            error_lessons=error_lessons,
        )

        content = super().run(payload)
        if self._has_contract(content):
            return LLMResponse(content=content, reasoning_content="")

        # One bounded formatting retry (B21 contract).
        retry_payload = (
            payload
            + "\n\nFORMAT REQUIRED:\n"
            + "<verdict>...</verdict>\n"
            + "<solution>...</solution>\n"
            + "If needed, include one or more <lemma>...</lemma> inside <solution>."
        )
        retry_content = super().run(retry_payload)
        return LLMResponse(content=retry_content, reasoning_content="")
