"""封装 DeepSeek / 火山方舟 API（兼容 OpenAI SDK），支持思考模式、工具调用与流式输出。"""

import json
import sys
from dataclasses import dataclass, field
from typing import Callable

import httpx
from openai import OpenAI

# 哨兵对象：区分「未传 stream_file（默认输出到 stderr）」和「主动传 None（禁用输出）」
_UNSET = object()


def _is_valid_json(s: str) -> bool:
    """判断字符串是否是合法 JSON。"""
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


@dataclass
class LLMResponse:
    """封装 DeepSeek API 响应，分离 reasoning_content 和 content。"""

    content: str  # 最终回答（解析器 Task 2.1-2.3 的输入源）
    reasoning_content: str = ""  # 思维链（思考模式下非空）
    tool_calls_trace: list[dict] = field(default_factory=list)  # 工具调用链路追踪


class LLMClient:
    """DeepSeek / 火山方舟 API 客户端，兼容 OpenAI SDK，支持流式输出。"""

    def __init__(self, config: dict, stream_file=_UNSET):
        """从 config dict 初始化客户端。

        Args:
            config: 应包含 deepseek.api_key / deepseek.base_url / deepseek.model 等字段。
            stream_file: 流式 token 的实时输出目标。
                - 不传（默认）：写入 sys.stderr。
                - 传入文件对象（如 sys.stdout）：写入该文件。
                - 传入 None：禁用实时输出。
        """
        ds = config["deepseek"]
        self._client = OpenAI(
            api_key=ds["api_key"],
            base_url=ds["base_url"],
            timeout=httpx.Timeout(1200.0, connect=30.0),  # 总超时 1200s，连接超时 30s
        )
        self._model = ds.get("model", "deepseek-chat")
        self._thinking = ds.get("thinking", False)
        self._max_tokens = ds.get("max_tokens", 16384)
        self._stream_file = sys.stderr if stream_file is _UNSET else stream_file

    # ------------------------------------------------------------------
    # 流式读取内部辅助
    # ------------------------------------------------------------------

    def _stream_completion(self, kwargs: dict) -> tuple[str, str, list[dict] | None]:
        """流式请求并实时输出，返回 (reasoning_content, content, tool_calls)。

        内置重试逻辑：对网络连接中断（RemoteProtocolError/ReadError）最多重试 2 次。
        """
        import time as _time

        last_error: Exception | None = None
        _MAX_STREAM_RETRIES = 2
        for _attempt in range(1 + _MAX_STREAM_RETRIES):
            if _attempt > 0:
                wait = 2 ** _attempt  # 2s, 4s 指数退避
                out = self._stream_file
                if out:
                    print(f"\n[RETRY {_attempt}/{_MAX_STREAM_RETRIES}] Network error: {last_error!r}. "
                          f"Retrying in {wait}s...", file=out, flush=True)
                else:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "LLM stream retry %d/%d after: %s", _attempt, _MAX_STREAM_RETRIES, last_error
                    )
                _time.sleep(wait)
            try:
                return self._do_stream_completion(kwargs)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                    httpx.ReadTimeout) as e:
                last_error = e
                continue
            # 其他异常直接抛出（不重试）
        raise last_error or RuntimeError("_stream_completion: exhausted retries")

    def _do_stream_completion(self, kwargs: dict) -> tuple[str, str, list[dict] | None]:
        """流式请求核心逻辑（不含重试）。"""
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)

        content = ""
        reasoning_content = ""
        tool_calls_data: dict[int, dict] = {}
        out = self._stream_file

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content += rc
                    if out:
                        print(rc, end="", flush=True, file=out)

                if delta.content:
                    content += delta.content
                    if out:
                        print(delta.content, end="", flush=True, file=out)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments
        except KeyboardInterrupt:
            # Windows 下 SSL 超时可能表现为 KeyboardInterrupt；返回已收集的部分内容
            if out:
                print("\n[WARN] 流式读取被中断，返回已收集内容", file=out, flush=True)
            # 清除不完整的工具调用（JSON 可能截断，无法安全执行）
            incomplete_keys = [
                k for k, d in tool_calls_data.items()
                if not _is_valid_json(d.get("arguments", ""))
            ]
            for k in incomplete_keys:
                del tool_calls_data[k]

        if out and (reasoning_content or content):
            print(file=out)

        tool_calls = None
        if tool_calls_data:
            tool_calls = [
                {"id": d["id"], "type": "function",
                 "function": {"name": d["name"], "arguments": d["arguments"]}}
                for _, d in sorted(tool_calls_data.items())
            ]

        return reasoning_content, content, tool_calls

    # ------------------------------------------------------------------
    # 内部辅助：构造请求 kwargs
    # ------------------------------------------------------------------

    def _build_kwargs(self, messages: list, thinking: bool | None = None, **extras) -> dict:
        """构造 API 请求参数字典，统一处理 thinking extra_body。"""
        use_thinking = self._thinking if thinking is None else thinking
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            **extras,
        }
        if use_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    # ------------------------------------------------------------------
    # 纯对话请求
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        thinking: bool | None = None,
    ) -> LLMResponse:
        """发送纯对话请求（流式），返回 LLMResponse。"""
        reasoning_content, content, _ = self._stream_completion(
            self._build_kwargs(messages, thinking=thinking)
        )
        return LLMResponse(
            content=content or "",
            reasoning_content=reasoning_content or "",
        )

    # ------------------------------------------------------------------
    # 思考模式 + 工具调用（单 turn 内多轮子请求）
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_tool_rounds: int = 10,
    ) -> LLMResponse:
        """思考模式下的多轮工具调用对话（流式）。"""
        trace: list[dict] = []
        last_reasoning = ""
        content = ""

        for _ in range(max_tool_rounds):
            kwargs = self._build_kwargs(messages, tools=tools)

            reasoning_content, content, tool_calls = self._stream_completion(kwargs)
            last_reasoning = reasoning_content or ""

            # 将 assistant 消息以 dict 追加（含 reasoning_content，DeepSeek 工具调用要求）
            assistant_msg: dict = {
                "role": "assistant",
                "content": content or None,
                "reasoning_content": reasoning_content or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls is None:
                break

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    func_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    # 工具调用参数不完整（流被截断），跳过该调用
                    continue
                result = tool_executor(func_name, func_args)
                trace.append({"name": func_name, "arguments": func_args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return LLMResponse(
            content=content or "",
            reasoning_content=last_reasoning,
            tool_calls_trace=trace,
        )

    # ------------------------------------------------------------------
    # 跨 turn 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def clear_reasoning_content(messages: list) -> None:
        """清除 messages 中所有 assistant 消息的 reasoning_content。

        在新 turn 开始前调用，节省带宽并避免传入历史思维链。
        兼容 SDK Message 对象和 dict 两种格式。
        """
        for message in messages:
            if isinstance(message, dict):
                if "reasoning_content" in message:
                    message["reasoning_content"] = None
            elif hasattr(message, "reasoning_content"):
                message.reasoning_content = None


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------


def create_llm_client(config: dict, stream_file=_UNSET) -> LLMClient:
    """根据 config['provider'] 创建对应的 LLMClient 实例。

    支持的 provider:
      - "deepseek" (默认): 使用 config["deepseek"] 配置
      - "volcano": 使用 config["volcano"] 配置（火山方舟引擎）

    stream_file: 传递给 LLMClient 的流式输出目标（语义同 LLMClient.__init__）。
    """
    provider = config.get("provider", "deepseek")
    # 未解析的环境变量占位符视为默认值 "deepseek"
    if not provider or provider.startswith("${"):
        provider = "deepseek"
    if provider == "volcano":
        return LLMClient({"deepseek": config["volcano"]}, stream_file=stream_file)
    if provider == "deepseek":
        return LLMClient(config, stream_file=stream_file)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'deepseek' or 'volcano'.")
