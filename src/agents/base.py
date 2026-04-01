"""BaseAgent runtime skeleton with stage-local memory and bounded tool loop."""

from __future__ import annotations

import json
from typing import Callable


class BaseAgent:
    """Minimal stage-scoped agent runtime.

    - Keeps stage-local message history in self.messages.
    - Resets memory at the start of each run.
    - Supports optional tool loop via llm_client.chat_with_tools with max_tool_rounds.
    """

    def __init__(
        self,
        *,
        llm_client,
        system_prompt: str,
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], str] | None = None,
        max_tool_rounds: int = 5,
        stream_prefix: str | None = None,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        self.stream_prefix = stream_prefix
        self.messages: list[dict] = []

    def reset_stage_memory(self) -> None:
        """Clear per-stage memory before a new stage run."""
        self.messages = []

    @staticmethod
    def _payload_to_text(payload: dict | str) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run(self, payload: dict | str) -> str:
        """Run one stage and return final assistant text."""
        self.reset_stage_memory()
        user_text = self._payload_to_text(payload)

        self.messages.append({"role": "system", "content": self.system_prompt})
        self.messages.append({"role": "user", "content": user_text})

        if self.tools:
            if self.tool_executor is None:
                raise ValueError("tool_executor is required when tools are configured")
            response = self.llm_client.chat_with_tools(
                self.messages,
                self.tools,
                self.tool_executor,
                max_tool_rounds=self.max_tool_rounds,
                stream_prefix=self.stream_prefix,
            )
        else:
            response = self.llm_client.chat(
                self.messages,
                thinking=True,
                stream_prefix=self.stream_prefix,
            )

        self.messages.append({
            "role": "assistant",
            "content": response.content or "",
            "reasoning_content": getattr(response, "reasoning_content", "") or "",
        })
        return response.content or ""
