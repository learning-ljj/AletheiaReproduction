"""BaseAgent runtime skeleton with stage-local memory and bounded tool loop."""

from __future__ import annotations

import json
from typing import Callable

from src.models.llm_client import LLMResponse


class BaseAgent:
    """Minimal stage-scoped agent runtime.

    - Keeps stage-local message history in self.messages.
    - Resets memory at the start of each run.
    - Supports optional tool loop via llm_client.chat_with_tools with max_rounds.
    """

    def __init__(
        self,
        *,
        llm_client,
        system_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], str] | None = None,
        max_rounds: int = 20,
        stream_prefix: str | None = None,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.max_rounds = max_rounds
        self.stream_prefix = stream_prefix
        self.messages: list[dict] = []

    def reset_stage_memory(self) -> None:
        """每次进入新阶段前清空阶段内消息历史。

        大白话：
        - 这是“单阶段短记忆”，不是全局长期记忆；
        - 不清空会把上一轮噪音带进下一轮，模型会越聊越乱。
        """
        self.messages = []

    @staticmethod
    def _payload_to_text(payload: dict | str) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run(self, payload: dict | str) -> LLMResponse:
        """执行单个阶段，并返回完整 LLMResponse。

        返回值不仅有 content，还保留 reasoning_content 与 tool_calls_trace。
        这样上层才能把“模型怎么想的、调了哪些工具”落盘审计。
        """
        try:
            user_text = self._payload_to_text(payload)

            self.messages.append({"role": "system", "content": self.system_prompt})
            self.messages.append({"role": "user", "content": user_text})

            if self.tools:
                if self.tool_executor is None:
                    raise ValueError("tool_executor is required when tools are configured")

                # 工具模式：模型可在一个阶段里反复调用工具，直到拿到足够证据。
                # chat_with_tools 返回的 tool_calls_trace 是“本次 run 的调用流水”，
                response = self.llm_client.chat_with_tools(
                    self.messages,
                    self.tools,
                    self.tool_executor,
                    max_rounds=self.max_rounds,
                    stream_prefix=self.stream_prefix,
                )
            else:
                # 纯对话模式：不调工具，直接让模型产出。
                response = self.llm_client.chat(
                    self.messages,
                    thinking=True,
                    stream_prefix=self.stream_prefix,
                )

            reasoning_trace = getattr(response, "reasoning_trace", []) or []
            last_reasoning = reasoning_trace[-1] if reasoning_trace else (getattr(response, "reasoning_content", "") or "")
            self.messages.append({
                "role": "assistant",
                "content": response.content or "",
                "reasoning_content": last_reasoning,
            })

            # 统一返回完整对象，保证上游随时能拿到 trace 做运行审计。
            return LLMResponse(
                content=response.content or "",
                reasoning_content=last_reasoning,
                reasoning_trace=reasoning_trace,
                tool_calls_trace=getattr(response, "tool_calls_trace", []) or [],
            )
        finally:
            # 关键约束：阶段结束后立刻清空该阶段消息缓存。
            # 这样下一轮调用时只带新输入，不会残留上次对话记忆。
            self.reset_stage_memory()
