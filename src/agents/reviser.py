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
        max_rounds: int = 20,
    ):
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            tool_executor=tool_executor,
            max_rounds=max_rounds,
            stream_prefix="REVISER",
        )

    @staticmethod
    def _build_input(
        problem_text: str,
        previous_solution: str,
        verification: str,
        lemma_context_items: list[str] | None = None,
    ) -> str:
        # Reviser 的输入是“同题目 + 旧答案 + verifier 报告”三件套。
        parts = [
            problem_text.strip()
            + "\n\n---\nPrevious Solution:\n"
            + (previous_solution or "").strip()
            + "\n\n---\nVerification Report:\n"
            + (verification or "").strip()
        ]
        if lemma_context_items:
            parts.append("\n\n---\nLemma Context:\n" + "\n".join(f"- {item}" for item in lemma_context_items))
        return "".join(parts)

    def run(
        self,
        *,
        problem_text: str,
        previous_solution: str,
        verification: str,
        lemma_context_items: list[str] | None = None,
    ) -> LLMResponse:

        payload = self._build_input(
            problem_text,
            previous_solution,
            verification,
            lemma_context_items=lemma_context_items,
        )
        return super().run(payload)
