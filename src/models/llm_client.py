"""封装 DeepSeek / 火山方舟 API（兼容 OpenAI SDK），支持思考模式、工具调用与流式输出。"""

import sys
from dataclasses import dataclass, field
from typing import Callable

from src.models.llm.message_sanitizer import MessageSanitizer
from src.models.llm.provider_factory import ProviderFactory
from src.models.llm.stream_transport import StreamTransport
from src.models.llm.tool_call_session import ToolCallSession

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
        normalized_config = config
        if "deepseek" not in normalized_config:
            normalized_config = ProviderFactory.resolve_provider_payload(config)
        bundle = ProviderFactory.build_client_bundle(normalized_config)
        self._client = bundle.client
        self._model = bundle.model
        self._thinking = bundle.thinking
        self._max_tokens = bundle.max_tokens
        self._stream_max_retries = bundle.stream_max_retries
        self._stream_retry_backoff_seconds = bundle.stream_retry_backoff_seconds
        self._stream_file = sys.stdout if stream_file is _UNSET else stream_file
        self._stream_transport = StreamTransport(
            client=self._client,
            stream_file=self._stream_file,
            max_retries=self._stream_max_retries,
            retry_backoff_seconds=self._stream_retry_backoff_seconds,
        )
        self._tool_call_session = ToolCallSession(
            stream_transport=self._stream_transport,
            build_kwargs=self._build_kwargs,
        )

    def _stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict] | None]:
        """Delegate stream read and retry logic to StreamTransport."""
        return self._stream_transport.stream_completion(
            kwargs,
            stream_prefix=stream_prefix,
        )

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
        # 核心来源说明：trace 由 ToolCallSession.run 在每次 tool_executor 调用后累积。
        # 大白话：每调用一次工具，就记一条 {name, arguments, result} 到 trace 里。
        # 这就是上层看到的“尝试轨迹”数据源。
        content, last_reasoning, trace = self._tool_call_session.run(
            messages=messages,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            stream_prefix=stream_prefix,
        )

        # 返回完整 LLMResponse，给上层（如 Generator）做重试合并和审计落盘。
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
        MessageSanitizer.clear_reasoning_content(messages)


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------


def create_llm_client(config: dict, stream_file=_UNSET) -> LLMClient:
    """根据配置选择 provider，并返回 LLMClient 实例。"""
    normalized_config = ProviderFactory.resolve_provider_payload(config)
    return LLMClient(normalized_config, stream_file=stream_file)
