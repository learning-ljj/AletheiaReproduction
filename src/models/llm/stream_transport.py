"""LLM 流式传输层：负责请求、分片解析与统一返回结构。"""

from __future__ import annotations

import sys
import time


class StreamTransport:
    """Handles low-level stream read, retry, and chunk parsing."""

    def __init__(
        self,
        *,
        client,
        stream_file,
        max_retries: int = 0,
    ):
        # 构造函数说明：
        # - client: 已构建的 SDK 客户端实例（用于发起流式请求）
        # - stream_file: 可选的输出文件/流（如 sys.stderr 或 open file），用于打印流式输出与重试信息
        # - max_retries: LLM API 调用失败时的最大重试次数（0 = 不重试），来自 settings.yaml resilience.llm_retry_max_attempts
        self._client = client
        self._stream_file = stream_file
        self._max_retries = max_retries

    def _log(
        self,
        message: str,
        *args,
    ) -> None:
        """将消息打印到 stream_file（或 stderr）并立即刷新。"""
        out = self._stream_file or sys.stderr
        if args:
            message = message % args
        print(message, file=out, flush=True)

    def stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict]]:
        """Request streaming completion with optional retry on any failure.

        重试策略：
        - 捕获任意 Exception，无论网络错误、API 返回错误还是其他异常都触发重试。
        - 每次重试前等待指数退避时间：backoff = 1 * (2 ** attempt) 秒。
        - 每次失败在 stream_file（或 stderr）上打印告警。
        - 超过 max_retries 次后仍失败，抛出最后一次异常。
        - KeyboardInterrupt / SystemExit 不会被捕获，保证可中断。
        """
        last_exc = None
        out = self._stream_file

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                # 指数退避：第 1 次重试等 2s，第 2 次等 4s，第 3 次等 8s …
                backoff = 1 * (2 ** (attempt - 1))
                self._log(
                    "\n[RETRY] 等待 %.1fs 后进行第 %d 次重试（最多 %d 次）...",
                    backoff, attempt, self._max_retries,
                )
                time.sleep(backoff)

            try:
                return self._do_stream_completion(kwargs, stream_prefix=stream_prefix)
            except Exception as exc:
                last_exc = exc
                self._log(
                    "\n[STREAM WARNING] API 请求失败（第 %d/%d 次）：%s: %s",
                    attempt + 1, self._max_retries + 1,
                    type(exc).__name__, exc,
                )
                # 还剩重试次数则继续循环，否则 fallthrough 到 raise

        # 所有重试均失败，抛出最后一次异常
        self._log("\n[STREAM ERROR] API 请求在 %d 次尝试后全部失败，放弃。", self._max_retries + 1)
        raise last_exc  # type: ignore[misc]  # 循环至少执行一次，last_exc 一定有值

    def _do_stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict]]:
        # 为了避免污染调用方传入的 kwargs，这里先复制一份再加 stream 开关。
        payload = dict(kwargs)
        payload["stream"] = True
        stream = self._client.chat.completions.create(**payload)

        # 初始化汇总变量
        content = ""
        reasoning_content = ""
        # 使用 dict 收集按 index 的工具调用片段，最后按索引排序合并成完整调用
        tool_calls_data: dict[int, dict] = {}
        out = self._stream_file

        # 如果传入 stream_prefix，则在输出流前打印前缀（便于多路并行输出识别）
        if out and stream_prefix:
            print(f"[{stream_prefix}] ", end="", flush=True, file=out)

        # 逐 chunk 遍历流式返回的数据，解析各字段并拼接
        for chunk in stream:
            # 有些 chunk 可能为空（无 choices），直接跳过
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 解析 reasoning_content（模型的内部思路片段）并打印到 out
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_content += rc
                if out:
                    print(rc, end="", flush=True, file=out)

            # 解析正文 content 并打印到 out
            if delta.content:
                content += delta.content
                if out:
                    print(delta.content, end="", flush=True, file=out)

            # 解析流式的工具调用片段（可能被拆分到多个 chunk）
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        # name/arguments 可能被流式拆分，多次追加以拼接完整字符串
                        if tc.function.name:
                            tool_calls_data[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["arguments"] += tc.function.arguments

        # 如果有输出流并且我们已经打印了内容，打印换行将行尾清理干净
        if out and (reasoning_content or content):
            print(file=out)

        # 将按索引收集的工具调用片段按顺序组装为最终的 list，保证上层一致地接收 list
        tool_calls = [
            {
                "id": data["id"],
                "type": "function",
                "function": {"name": data["name"], "arguments": data["arguments"]},
            }
            for _, data in sorted(tool_calls_data.items())
        ]

        return reasoning_content, content, tool_calls
