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
        max_tool_rounds: int = 20,
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
        lemma_context_items: list[str] | None = None,
        error_lessons: str | None = None,
    ) -> str:
        # 输入拼装策略：
        # - 先给题目本体；
        # - 再给工具使用规则；
        # - 最后附历史摘要与错误经验。
        # 这样模型拿到的是“问题 + 约束 + 复盘上下文”的组合输入。
        parts = [problem_text.strip()]
        # parts.append(
        #     "\n\nTooling Hint:\n"
        #     "- If external references are needed, call tool `call_searcher` with a focused query.\n"
        #     "- After retrieval, cite artifact paths as [cite:path]."
        # )
        if lemma_context_items:
            parts.append("\n\n---\nLemma Context:\n" + "\n".join(f"- {item}" for item in lemma_context_items))
        if error_lessons:
            parts.append("\n\n---\nError Lessons:\n" + error_lessons.strip())
        return "".join(parts)

    def run(
        self,
        *,
        problem_text: str,
        lemma_context_items: list[str] | None = None,
        error_lessons: str | None = None,
    ) -> LLMResponse:
        # 仅执行一次生成。
        # 大白话：Generator 不再负责格式兜底重试。
        # 候选解答里的格式问题统一交给 Verifier 判定，再由 Reviser 修复。
        payload = self._build_input(
            problem_text,
            lemma_context_items=lemma_context_items,
            error_lessons=error_lessons,
        )
        return super().run(payload)
