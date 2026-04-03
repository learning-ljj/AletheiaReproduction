"""Streaming transport layer for LLM chat completions."""

from __future__ import annotations

import logging

import httpx


class StreamTransport:
    """Handles low-level stream read, retry, and chunk parsing."""

    def __init__(
        self,
        *,
        client,
        stream_file,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
    ):
        self._client = client
        self._stream_file = stream_file
        self._stream_max_retries = max(0, int(max_retries))
        self._stream_retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    @staticmethod
    def _raise_stream_failure(last_error: Exception | None) -> None:
        if last_error is None:
            raise RuntimeError("LLM stream failed without an underlying error")
        if isinstance(last_error, httpx.TimeoutException):
            raise TimeoutError("LLM stream timed out after retries") from last_error
        if isinstance(last_error, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
            raise ConnectionError("LLM stream failed after retries") from last_error
        raise RuntimeError("LLM stream failed after retries") from last_error

    def stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict] | None]:
        """Request streaming completion and return reasoning/content/tool-calls."""
        import time as _time

        last_error: Exception | None = None
        for attempt in range(1 + self._stream_max_retries):
            if attempt > 0:
                wait = self._stream_retry_backoff_seconds * (2 ** (attempt - 1))
                out = self._stream_file
                if out:
                    print(
                        f"\n[RETRY {attempt}/{self._stream_max_retries}] Network error: {last_error!r}. "
                        f"Retrying in {wait}s...",
                        file=out,
                        flush=True,
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "LLM stream retry %d/%d after: %s",
                        attempt,
                        self._stream_max_retries,
                        last_error,
                    )
                _time.sleep(wait)
            try:
                return self._do_stream_completion(kwargs, stream_prefix=stream_prefix)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                continue

        self._raise_stream_failure(last_error)

    def _do_stream_completion(
        self,
        kwargs: dict,
        stream_prefix: str | None = None,
    ) -> tuple[str, str, list[dict] | None]:
        payload = dict(kwargs)
        payload["stream"] = True
        try:
            stream = self._client.chat.completions.create(**payload)
        except KeyboardInterrupt as exc:
            out = self._stream_file
            if out:
                print("\n[WARN] 建立流式连接中断，触发自动重试", file=out, flush=True)
            raise httpx.ReadTimeout("LLM stream interrupted before first chunk") from exc

        content = ""
        reasoning_content = ""
        tool_calls_data: dict[int, dict] = {}
        out = self._stream_file

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
            if out:
                print("\n[WARN] 流式读取中断，触发自动重试", file=out, flush=True)
            raise httpx.ReadTimeout("LLM stream interrupted") from exc

        if out and (reasoning_content or content):
            print(file=out)

        tool_calls = None
        if tool_calls_data:
            tool_calls = [
                {
                    "id": data["id"],
                    "type": "function",
                    "function": {"name": data["name"], "arguments": data["arguments"]},
                }
                for _, data in sorted(tool_calls_data.items())
            ]

        return reasoning_content, content, tool_calls
