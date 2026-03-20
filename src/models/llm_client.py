"""封装 DeepSeek / 火山方舟 API（兼容 OpenAI SDK），支持思考模式、工具调用与流式输出。"""

import json
import sys
from dataclasses import dataclass, field
from typing import Callable

import httpx
from openai import OpenAI

# 哨兵对象：区分「未传 stream_file（默认输出到 stdout）」和「主动传 None（禁用输出）」
_UNSET = object()


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
                - 不传（默认）：写入 sys.stdout。
                - 传入文件对象（如 sys.stdout）：写入该文件。
                - 传入 None：禁用实时输出。
        """
        ds = config["deepseek"]
        connect_timeout = float(ds.get("connect_timeout_seconds", 30.0))
        read_timeout = float(ds.get("read_timeout_seconds", 90.0))
        write_timeout = float(ds.get("write_timeout_seconds", read_timeout))
        pool_timeout = float(ds.get("pool_timeout_seconds", connect_timeout))
        self._client = OpenAI(
            api_key=ds["api_key"],
            base_url=ds["base_url"],
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )
        self._model = ds.get("model", "deepseek-chat")
        self._thinking = ds.get("thinking", False)
        self._max_tokens = ds.get("max_tokens", 16384)
        self._stream_max_retries = max(int(ds.get("stream_max_retries", 2)), 0)
        self._stream_retry_backoff_seconds = max(float(ds.get("stream_retry_backoff_seconds", 2.0)), 0.0)
        self._stream_file = sys.stdout if stream_file is _UNSET else stream_file

    @staticmethod
    def _raise_stream_failure(last_error: Exception | None) -> None:
        """把底层网络异常映射为上层可分类的运行时错误。"""
        if last_error is None:
            raise RuntimeError("LLM stream failed without an underlying error")
        if isinstance(last_error, httpx.TimeoutException):
            raise TimeoutError("LLM stream timed out after retries") from last_error
        if isinstance(last_error, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
            raise ConnectionError("LLM stream failed after retries") from last_error
        raise RuntimeError("LLM stream failed after retries") from last_error

    # ------------------------------------------------------------------
    # 流式读取内部辅助
    # ------------------------------------------------------------------

    def _stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict] | None]:
        """流式请求并实时输出，返回 (reasoning_content, content, tool_calls)。

        内置重试逻辑：对网络连接中断（RemoteProtocolError/ReadError）最多重试 2 次。
        """
        import time as _time

        last_error: Exception | None = None
        for _attempt in range(1 + self._stream_max_retries):
            if _attempt > 0:
                wait = self._stream_retry_backoff_seconds * (2 ** (_attempt - 1))
                out = self._stream_file
                if out:
                    print(f"\n[RETRY {_attempt}/{self._stream_max_retries}] Network error: {last_error!r}. "
                          f"Retrying in {wait}s...", file=out, flush=True)
                else:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "LLM stream retry %d/%d after: %s", _attempt, self._stream_max_retries, last_error
                    )
                _time.sleep(wait)
            try:
                return self._do_stream_completion(kwargs, stream_prefix=stream_prefix)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                    httpx.TimeoutException) as e:
                last_error = e
                continue
            # 其他异常直接抛出（不重试）
        self._raise_stream_failure(last_error)

    def _do_stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict] | None]:
        """流式请求核心逻辑（不含重试）。"""
        kwargs["stream"] = True
        try:
            stream = self._client.chat.completions.create(**kwargs)
        except KeyboardInterrupt as exc:
            # 修复：建连阶段也可能抛 KeyboardInterrupt（Windows/代理链路抖动）。
            # 统一转换为可重试异常，交给 _stream_completion 指数退避处理。
            out = self._stream_file
            if out:
                print("\n[WARN] 建立流式连接中断，触发自动重试", file=out, flush=True)
            raise httpx.ReadTimeout("LLM stream interrupted before first chunk") from exc

        content = ""
        reasoning_content = ""
        tool_calls_data: dict[int, dict] = {}
        out = self._stream_file

        # 每次调用在流首加节点前缀，便于终端实时区分阶段来源。
        if out and stream_prefix:
            print(f"[{stream_prefix}] ", end="", flush=True, file=out)

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
        except KeyboardInterrupt as exc:
            # Windows 下网络抖动有时会表现为 KeyboardInterrupt；
            # 不返回截断内容，改为交给上层重试，避免 content 为空时误入下一阶段。
            if out:
                print("\n[WARN] 流式读取中断，触发自动重试", file=out, flush=True)
            raise httpx.ReadTimeout("LLM stream interrupted") from exc

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
        stream_prefix: str | None = None,
    ) -> LLMResponse:
        """发送纯对话请求（流式），返回 LLMResponse。"""
        reasoning_content, content, _ = self._stream_completion(
            self._build_kwargs(messages, thinking=thinking),
            stream_prefix=stream_prefix,
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
        stream_prefix: str | None = None,
    ) -> LLMResponse:
        """思考模式下的多轮工具调用对话（流式）。"""
        trace: list[dict] = []
        last_reasoning = ""
        content = ""

        for _ in range(max_tool_rounds):
            kwargs = self._build_kwargs(messages, tools=tools)

            reasoning_content, content, tool_calls = self._stream_completion(
                kwargs,
                stream_prefix=stream_prefix,
            )
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
    def _configured(value: str | None) -> bool:
        return bool(value) and (not str(value).startswith("${"))

    def _contains_placeholder(value: str | None) -> bool:
        return isinstance(value, str) and "${" in value

    def _resolve_provider_config(provider_name: str) -> dict:
        shared_defaults = config.get("llm_defaults") or {}
        provider_config = config.get(provider_name) or {}
        return {**shared_defaults, **provider_config}

    provider = config.get("provider")
    # 若 provider 未配置或仍是占位符，则按可用密钥自动选择：volcano 优先，deepseek 备选
    if not _configured(provider):
        volcano_api_key = (config.get("volcano") or {}).get("api_key")
        deepseek_api_key = (config.get("deepseek") or {}).get("api_key")
        if _configured(volcano_api_key):
            provider = "volcano"
        elif _configured(deepseek_api_key):
            provider = "deepseek"
        else:
            provider = "deepseek"
    # 在返回客户端前做额外校验，给出更明确的报错（常见原因：.env 未加载、占位符未替换、base_url 格式错误）
    if provider == "volcano":
        volcano_cfg = _resolve_provider_config("volcano")
        api_key = volcano_cfg.get("api_key")
        base_url = volcano_cfg.get("base_url")
        if not api_key or _contains_placeholder(api_key):
            raise ValueError(
                "Volcano provider selected but `volcano.api_key` is missing or contains placeholder. "
                "Ensure .env contains VOLCANO_API_KEY and that you called load_dotenv() before loading config."
            )
        if base_url and not str(base_url).startswith("http"):
            raise ValueError(
                f"Volcano base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
            )
        return LLMClient({"deepseek": volcano_cfg}, stream_file=stream_file)
    if provider == "deepseek":
        deep_cfg = _resolve_provider_config("deepseek")
        api_key = deep_cfg.get("api_key")
        base_url = deep_cfg.get("base_url")
        if not api_key or _contains_placeholder(api_key):
            raise ValueError(
                "DeepSeek provider selected but `deepseek.api_key` is missing or contains placeholder. "
                "Ensure .env contains DEEPSEEK_API_KEY and that you called load_dotenv() before loading config."
            )
        if base_url and not str(base_url).startswith("http"):
            raise ValueError(
                f"DeepSeek base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
            )
        return LLMClient({"deepseek": deep_cfg}, stream_file=stream_file)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'deepseek' or 'volcano'.")
