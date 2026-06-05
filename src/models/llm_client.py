"""封装 provider（例如 DeepSeek / 火山方舟）API，兼容 OpenAI SDK，支持思考模式、工具调用与流式输出。"""

import sys
from dataclasses import dataclass, field
from typing import Callable

from src.models.llm.provider_factory import ProviderFactory
from src.models.llm.stream_transport import StreamTransport
from src.models.llm.tool_call_session import ToolCallSession

# 哨兵对象：区分「未传 stream_file（默认输出到 stdout）」和「主动传 None（禁用输出）」
_UNSET = object()


@dataclass
class LLMResponse:
    """封装统一 LLM 响应，分离 reasoning_content 和 content。"""

    content: str  # 最终回答（解析器 Task 2.1-2.3 的输入源）
    reasoning_content: str = ""  # 思维链（思考模式下非空）
    reasoning_trace: list[str] = field(default_factory=list)  # 单次工具循环内的思维链列表
    tool_calls_trace: list[dict] = field(default_factory=list)  # 工具调用链路追踪


class LLMClient:
    """Provider API 客户端，兼容 OpenAI SDK，支持流式输出。"""

    def __init__(self, config: dict, stream_file=_UNSET):
        """从 config dict 初始化客户端。

        Args:
            config: 已归一化的 provider payload（必须包含 `provider` 键），
                通常由 `ProviderFactory.resolve_provider_payload(config)` 生成。
            stream_file: 流式 token 的实时输出目标。
                - 不传（默认）：写入 sys.stdout。
                - 传入文件对象（如 sys.stdout）：写入该文件。
                - 传入 None：禁用实时输出。
        """
        # 要求 caller 传入已归一化 payload（含 'provider' 键）。
        bundle = ProviderFactory.build_client_bundle(config)
        self._client = bundle.client
        self._model = bundle.model
        self._thinking = bundle.thinking
        self._max_tokens = bundle.max_tokens
        self._stream_file = sys.stdout if stream_file is _UNSET else stream_file

        # 从归一化 payload 中提取弹性配置
        # llm_retry_max_attempts 由 ProviderFactory.resolve_provider_payload 注入
        self._llm_retry_max_attempts = int(config.get("llm_retry_max_attempts", 0))

        # 职责拆分：
        # StreamTransport 负责底层流式读取与网络重试。
        # ToolCallSession 负责“单轮对话内”多次工具调用闭环。
        self._stream_transport = StreamTransport(
            client=self._client,
            stream_file=self._stream_file,
            max_retries=self._llm_retry_max_attempts,
        )
        self._tool_call_session = ToolCallSession(
            stream_transport=self._stream_transport,
            build_kwargs=self._build_kwargs,
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
        reasoning_content, content, _ = self._stream_transport.stream_completion(
            self._build_kwargs(messages, thinking=thinking),
            stream_prefix=stream_prefix,
        )
        return LLMResponse(
            content=content or "",
            reasoning_content=reasoning_content or "",
            reasoning_trace=[reasoning_content] if reasoning_content else [],
        )

    # ------------------------------------------------------------------
    # 思考模式 + 工具调用（单 turn 内多轮子请求）
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 20,
        stream_prefix: str | None = None,
    ) -> LLMResponse:
        """思考模式下的多轮工具调用对话（流式）。"""
        # 核心来源说明：trace 由 ToolCallSession.run 在每次 tool_executor 调用后累积。
        # 大白话：每调用一次工具，就记一条 {name, arguments, result} 到 trace 里。
        # 这就是上层看到的“尝试轨迹”数据源。
        content, last_reasoning, trace, reasoning_trace = self._tool_call_session.run(
            messages=messages,
            tools=tools,
            tool_executor=tool_executor,
            max_rounds=max_rounds,
            stream_prefix=stream_prefix,
        )

        # 返回完整 LLMResponse，给上层（如 Generator）做重试合并和审计落盘。
        return LLMResponse(
            content=content or "",
            reasoning_content=last_reasoning,
            reasoning_trace=reasoning_trace,
            tool_calls_trace=trace,
        )

    # ------------------------------------------------------------------
    # 跨 turn ：清除 reasoning_content
    # ------------------------------------------------------------------

    @staticmethod
    def clear_reasoning_content(messages: list[dict]) -> None:
        """清除 messages 中所有 assistant 消息的 reasoning_content。

        在新 turn 开始前调用，避免传入历史思维链。
        当前主链仅处理 dict 形式消息。
        """
        for message in messages:
            if isinstance(message, dict) and "reasoning_content" in message:
                message["reasoning_content"] = None


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------


def create_llm_client(config: dict, stream_file=_UNSET) -> LLMClient:
    """根据配置选择 provider，并返回 LLMClient 实例。"""
    normalized_config = ProviderFactory.resolve_provider_payload(config)
    return LLMClient(normalized_config, stream_file=stream_file)
