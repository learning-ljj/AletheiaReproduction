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
        max_rounds: int = 20,
    ):
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            tool_executor=tool_executor,
            max_rounds=max_rounds,
            stream_prefix="GENERATOR",
        )

    @staticmethod
    def _build_input(
        problem_text: str,
        *,
        lemma_context_items: list[str] | None = None,
        verification: str | None = None,
    ) -> str:
        # 输入：题目、历史摘要与上一轮次错误经验。
        parts = [problem_text.strip()]
        if lemma_context_items:
            parts.append("\n\n---\nLemma Context:\n" + "\n".join(f"- {item}" for item in lemma_context_items))
        if verification:
            parts.append("\n\n---\nVerification:\n" + verification.strip())
        return "".join(parts)

    def run(
        self,
        *,
        problem_text: str,
        lemma_context_items: list[str] | None = None,
        verification: str | None = None,
    ) -> LLMResponse:
        
        payload = self._build_input(
            problem_text,
            lemma_context_items=lemma_context_items,
            verification=verification,
        )
        return super().run(payload)
