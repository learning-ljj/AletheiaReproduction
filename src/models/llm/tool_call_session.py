"""单个 assistant 回合内的工具调用会话循环。"""

from __future__ import annotations

import json
import logging
from typing import Callable

from src.models.llm.stream_transport import StreamTransport


class ToolCallSession:
    """Runs the bounded multi-round tool loop for chat_with_tools."""

    def __init__(
        self,
        *,
        stream_transport: StreamTransport,
        build_kwargs: Callable[..., dict],
    ):
        # 构造器说明：
        # stream_transport: 负责与 LLM 发起流式请求并解析流式断片，返回 (reasoning_content, content, tool_calls)
        # build_kwargs: 可调用对象，用于根据当前 messages/tools 构建传给 LLM 的请求参数字典
        self._stream_transport: StreamTransport = stream_transport
        self._build_kwargs = build_kwargs

    def run(
        self,
        *,
        messages: list,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 20,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict], list[str]]:
        """Return (content, last_reasoning, tool_trace, reasoning_trace).

        约束：stream_transport 始终返回 list 类型 tool_calls，
        因此本循环以空列表表示“本轮无工具调用意图”。
        """
        trace: list[dict] = []
        reasoning_trace: list[str] = []
        last_reasoning = ""
        content = ""

        # 单个阶段内的“工具闭环”最多跑 max_rounds 轮。
        for _ in range(max_rounds):
            # 根据当前对话历史与可用工具构建本轮 LLM 请求参数,包装为字典 kwargs
            kwargs = self._build_kwargs(messages, tools=tools)
            # 发起流式请求并解析出：思路片段、正文、以及本轮意图触发的工具调用列表
            reasoning_content, content, tool_calls = self._stream_transport.stream_completion(
                kwargs,
                stream_prefix=stream_prefix,
            )
            last_reasoning = reasoning_content or ""
            reasoning_trace.append(last_reasoning)

            # 将 assistant 的输出写入 messages，供下一轮模型看到自己的话
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "reasoning_content": reasoning_content or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # 如果模型本轮未发出任何工具调用意图，则会话结束
            if not tool_calls:
                break

            # 否则遍历每个工具调用，尝试解析参数并执行工具
            for tool_call in tool_calls:
                # OpenAI function-calling 约定：工具名在 function.name，参数在 function.arguments(JSON 字符串)
                func_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                try:
                    # 参数期望为 JSON 字符串；若流式被截断导致无效 JSON，则跳过该调用
                    func_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    # 流式截断导致参数不完整时，给出警告并跳过该次调用。
                    logging.getLogger(__name__).warning(
                        "Skip tool call due to invalid/incomplete JSON args. function=%s raw_args=%r",
                        func_name,
                        raw_args,
                    )
                    continue

                # 真正执行工具，并记录审计轨迹，记录了调用名、入参、返回值。
                result = tool_executor(func_name, func_args)
                trace.append({"name": func_name, "arguments": func_args, "result": result})

                # 把工具结果回填到对话历史，供下一轮模型继续推理。
                # 模型看到的是一条 role=tool 的文本反馈，而不是原始 Python 对象。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

        # 返回最后一轮正文 + 最后一轮思维链 + 全部工具调用轨迹 + 全部思维链列表。
        # 这里 trace 是“本次 run 内完整轨迹”，上层可继续做跨尝试 merge。
        return content or "", last_reasoning, trace, reasoning_trace
