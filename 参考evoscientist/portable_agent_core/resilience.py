"""Resilience helpers inspired by EvoScientist's error handling ideas.

Key ideas borrowed from the source repository:
1. Tool failures should become structured results instead of hard crashes.
2. Retries should be explicit and bounded.
3. Self-correction should be a loop with validation, not a blind retry.

中文说明：
该模块封装了常用的容错模式：有界重试、将异常转换为可序列化的结果对象、
以及带有校验与修复策略的自我修正循环（self-correction），便于上层将错误
视为可处理的状态而非直接崩溃。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class RetryPolicy:
    # Max attempts includes the first try.
    # 最大尝试次数（包含第一次尝试）。
    max_attempts: int = 3
    # Base delay is multiplied by the backoff factor after each failure.
    # 基本延迟（秒），每次失败后按指数回退增长。
    base_delay_seconds: float = 0.5
    # Backoff factor controls exponential growth.
    # 回退因子，用于计算下一次重试的等待时间。
    backoff_factor: float = 2.0
    # Allowed exception types keep retries selective.
    # 可重试的异常类型元组，仅对这些异常进行重试以避免不必要的重试。
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )


@dataclass(slots=True)
class ToolExecutionResult:
    # Tool or step name.
    # 名称：对应工具或步骤的标识。
    name: str
    # Success is false when an exception was caught.
    # success 标志表示操作是否成功（False 表示捕获到异常或质量校验失败）。
    success: bool
    # Content is the model-friendly message or normal result body.
    # content：供模型或界面显示的友好文本，可以是结果或错误说明。
    content: str
    # Raw value is preserved for callers that need structured access.
    # value：原始返回值，供需要结构化数据的调用方使用。
    value: Any = None
    # Attempts help debugging flaky providers.
    # attempts：尝试次数统计，便于调试波动性提供者。
    attempts: int = 1
    # Error text is separate from content for programmatic use.
    # error：程序化错误文本（通常是 exception 的字符串形式）。
    error: str = ""


@dataclass(slots=True)
class ValidationIssue:
    # Short machine-friendly code such as "empty_results".
    # code：简短的机器可识别错误码，例如 "empty_results"。
    code: str
    # Human-friendly explanation.
    # message：面向人的错误解释，用于日志或展示。
    message: str


async def retry_async(
    operation: Callable[[], Awaitable[Any]],
    policy: RetryPolicy,
) -> Any:
    """Run an async operation with bounded exponential backoff."""

    # Delay starts at the base value and grows after each failed attempt.
    delay = policy.base_delay_seconds
    # The last exception is kept so callers still get the real root cause.
    last_error: Exception | None = None

    # Attempt numbers start at 1 because that is easier to reason about.
    for attempt in range(1, policy.max_attempts + 1):
        try:
            # Return as soon as one attempt succeeds.
            return await operation()
        except policy.retryable_exceptions as exc:
            # Store the latest transient failure for a final re-raise.
            last_error = exc
            # If the last allowed attempt failed, stop retrying.
            if attempt >= policy.max_attempts:
                break
            # Sleep before retrying so a flaky network or provider can recover.
            await asyncio.sleep(delay)
            # Increase the next delay using exponential backoff.
            delay *= policy.backoff_factor

    # Re-raise the last captured exception if every attempt failed.
    if last_error is not None:
        raise last_error

    # This line should never be hit, but it keeps the function total.
    raise RuntimeError("retry_async reached an unexpected terminal state")


async def guarded_tool_call(
    name: str,
    operation: Callable[[], Awaitable[Any]],
    policy: RetryPolicy | None = None,
) -> ToolExecutionResult:
    """Execute a tool-like operation and convert exceptions into data."""

    # Use a default policy when the caller does not provide one.
    retry_policy = policy or RetryPolicy()

    try:
        # Route the operation through retry_async to handle transient failures.
        value = await retry_async(operation, retry_policy)
        # Convert non-string outputs into strings only for display, not storage.
        content = value if isinstance(value, str) else repr(value)
        # Return a success object instead of leaking framework-specific messages.
        return ToolExecutionResult(
            name=name,
            success=True,
            content=content,
            value=value,
            attempts=retry_policy.max_attempts,
        )
    except Exception as exc:  # noqa: BLE001
        # Produce a model-friendly error message inspired by EvoScientist.
        error_message = (
            f"[TOOL ERROR] {name} failed with {type(exc).__name__}: {exc}. "
            "You may retry with a smaller scope, switch provider, or ask the user "
            "for clarification."
        )
        return ToolExecutionResult(
            name=name,
            success=False,
            content=error_message,
            value=None,
            attempts=retry_policy.max_attempts,
            error=str(exc),
        )


async def run_with_self_correction(
    name: str,
    initial_input: dict[str, Any],
    step: Callable[[dict[str, Any]], Awaitable[Any]],
    validate: Callable[[Any], ValidationIssue | None],
    repair: Callable[[dict[str, Any], ValidationIssue], dict[str, Any] | None],
    max_corrections: int = 2,
) -> ToolExecutionResult:
    """Run a step, validate the output, and repair the input when needed."""

    # Current input may change after each validation failure.
    current_input = dict(initial_input)

    # The loop is bounded so self-correction cannot recurse forever.
    for correction_round in range(max_corrections + 1):
        # Wrap the current step execution so exceptions are still converted to data.
        result = await guarded_tool_call(
            name=name,
            operation=lambda: step(current_input),
        )

        # If the tool itself failed, stop here and return the guarded error object.
        if not result.success:
            return result

        # Run quality validation only on successful step outputs.
        issue = validate(result.value)

        # No issue means the output is good enough to hand off.
        if issue is None:
            result.attempts = correction_round + 1
            return result

        # If we already spent all correction rounds, return the latest result.
        if correction_round >= max_corrections:
            result.success = False
            result.error = issue.message
            result.content = (
                f"[QUALITY ERROR] {name} produced an unusable result: {issue.message}. "
                "Stop the loop and surface the problem to the main agent."
            )
            result.attempts = correction_round + 1
            return result

        # Ask the repair function to rewrite the next input.
        next_input = repair(current_input, issue)

        # If the repair function cannot improve the input, stop early.
        if next_input is None:
            result.success = False
            result.error = issue.message
            result.content = (
                f"[QUALITY ERROR] {name} failed validation and no repair path exists: "
                f"{issue.message}"
            )
            result.attempts = correction_round + 1
            return result

        # Replace the current input and let the loop try again.
        current_input = next_input

    # The loop is total, so reaching this line would be a bug.
    raise RuntimeError("run_with_self_correction exhausted unexpectedly")
