"""Tool-calling session loop for one assistant turn."""

from __future__ import annotations

import json
from typing import Callable


class ToolCallSession:
    """Runs the bounded multi-round tool loop for chat_with_tools."""

    def __init__(
        self,
        *,
        stream_transport,
        build_kwargs: Callable[..., dict],
    ):
        self._stream_transport = stream_transport
        self._build_kwargs = build_kwargs

    def run(
        self,
        *,
        messages: list,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_tool_rounds: int = 10,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict]]:
        """Return (content, reasoning_content, tool_trace)."""
        trace: list[dict] = []
        last_reasoning = ""
        content = ""

        # 单个阶段内的“工具闭环”最多跑 max_tool_rounds 轮。
        # 大白话：每一轮都可能是“模型说一句 -> 决定要不要调工具 -> 工具结果喂回去”。
        # 跑到上限就停，防止模型陷入无限工具循环。
        for _ in range(max_tool_rounds):
            kwargs = self._build_kwargs(messages, tools=tools)
            reasoning_content, content, tool_calls = self._stream_transport.stream_completion(
                kwargs,
                stream_prefix=stream_prefix,
            )
            last_reasoning = reasoning_content or ""

            # 把这一轮 assistant 输出先落到 messages。
            # 大白话：后续如果还要继续调工具，模型下一轮必须“看见自己刚才说过什么”。
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "reasoning_content": reasoning_content or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # 约定：tool_calls is None 表示模型本轮没有工具意图，直接结束会话。
            # 注意这里区分 None 与 []，因为上游流式解析会用 None 代表“无工具分支”。
            if tool_calls is None:
                break

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                try:
                    func_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    # 流式截断导致参数不完整时，跳过该次调用。
                    continue

                # 真正执行工具，并记录审计轨迹。
                # 大白话：trace 就是“工具调用流水账”，记录了调用名、入参、返回值。
                # Generator 的“两次尝试 merge”里，合并的就是这个 trace。
                result = tool_executor(func_name, func_args)
                trace.append({"name": func_name, "arguments": func_args, "result": result})

                # 把工具结果回填到对话历史，供下一轮模型继续推理。
                # 大白话：模型不是直接拿 Python 对象，而是拿到一条 role=tool 的文本反馈。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

        # 返回最后一轮正文 + 最后一轮思维链 + 全部工具调用轨迹。
        # 这里 trace 是“本次 run 内完整轨迹”，上层可继续做跨尝试 merge。
        return content or "", last_reasoning, trace
