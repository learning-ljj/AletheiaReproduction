"""工具注册表：OpenAI Function Calling schema 与统一执行分发。"""

from typing import Callable

from src.agents.citation_reviewer import CitationReviewerAgent
from src.agents.searcher import SearcherAgent
from src.memory.problem_memory import get_current_problem_memory
from src.tools.artifact_reader import read_artifact_layer
from src.tools.code_executor import run_python
from src.tools.envelope import (
    format_tool_error,
    format_tool_success,
)
from src.tools.schemas import get_tool_schemas as _schema_get_tool_schemas


_SEARCH_SOURCE_HANDLERS: dict[str, Callable[[str, int], list[dict]]] = {}


def configure_tool_resilience(*, max_attempts: int = 1, backoff_seconds: float = 0.0) -> None:
    """Compatibility hook for settings; MVP middleware keeps single-attempt execution."""
    _ = max_attempts
    _ = backoff_seconds


def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _format_run_python(code: str) -> str:
    """执行代码并返回统一成功包络。"""
    result = run_python(code)
    parts = []
    if result["stdout"]:
        parts.append(f"stdout:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"stderr:\n{result['stderr']}")
    if not parts:
        parts.append("(no output)")
    parts.append(f"exit_code: {result['exit_code']}")
    rendered = "\n".join(parts)
    return format_tool_success(
        tool="run_python",
        data={
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", 1),
            "rendered": rendered,
        },
    )


def _format_call_searcher(
    query: str | None = None,
    query_bundle: list[str] | None = None,
    **extra_args,
) -> str:
    """调用 SearcherAgent 并返回检索摘要与落盘路径。"""
    problem_memory = get_current_problem_memory()
    if problem_memory is None:
        return format_tool_error(
            tool="call_searcher",
            error_code="NO_PROBLEM_MEMORY",
            message="ProblemMemory context is missing for call_searcher.",
            retryable=False,
        )
    if not _SEARCH_SOURCE_HANDLERS:
        return format_tool_error(
            tool="call_searcher",
            error_code="NO_SEARCH_SOURCES_CONFIGURED",
            message="Searcher source handlers are not configured.",
            retryable=False,
        )

    agent = SearcherAgent(
        problem_memory=problem_memory,
        source_handlers=_SEARCH_SOURCE_HANDLERS,
    )

    # 重点说明：agent.run 不仅会返回 papers，也会返回 errors/recovered_errors。
    # 这些错误字段会原样回传给 LLM，帮助模型决定下一步动作，而不是像传统流程那样“空结果即静默降级”。
    result = agent.run(query=query, query_bundle=query_bundle)
    has_errors = bool(result.get("has_errors"))

    payload: dict = {
        "query": query,
        "query_bundle": query_bundle or [],
        "paper_count": result.get("count", 0),
        "stages": result.get("stages", {}),
        "papers": result.get("papers", []),
        "has_errors": has_errors,
        "error_count": int(result.get("error_count", 0) or 0),
        "errors": result.get("errors", []),
        "recovered_errors": result.get("recovered_errors", []),
        "llm_action_hint": result.get("llm_action_hint", ""),
    }
    if extra_args:
        payload["extra_args"] = extra_args
    return format_tool_success(tool="call_searcher", data=payload)


def _format_read_artifact_layer(path: str, layer: int) -> str:
    """按层读取 artifact 文档（artifact_reader 内部已统一包络）。"""
    return read_artifact_layer(path=path, layer=layer)


def _format_call_citation_reviewer(
    cites: list[str] | None = None,
    claim_spans: list[str] | None = None,
    **extra_args,
) -> str:
    """调用 CitationReviewerAgent 并返回逐条引用审查结果。"""
    problem_memory = get_current_problem_memory()
    if problem_memory is None:
        return format_tool_error(
            tool="call_citation_reviewer",
            error_code="NO_PROBLEM_MEMORY",
            message="ProblemMemory context is missing for call_citation_reviewer.",
            retryable=False,
        )

    normalized_cites = [(item or "").strip() for item in (cites or []) if (item or "").strip()]
    normalized_spans = [str(item or "").strip() for item in (claim_spans or [])]

    reviewer = CitationReviewerAgent(problem_memory=problem_memory)
    review = reviewer.review(cites=normalized_cites, claim_spans=normalized_spans)

    payload: dict = {
        "cites": normalized_cites,
        "claim_spans": normalized_spans,
        "summary": review.get("summary", ""),
        "items": review.get("items", []),
        "fail_count": review.get("fail_count", 0),
        "severity_suggestion": review.get("severity_suggestion", "MINOR_FLAW"),
    }
    if extra_args:
        payload["extra_args"] = extra_args
    return format_tool_success(tool="call_citation_reviewer", data=payload)


# 函数名 → 可调用对象的映射
_TOOL_MAP: dict = {
    "run_python": _format_run_python,
    "call_searcher": _format_call_searcher,
    "read_artifact_layer": _format_read_artifact_layer,
    "call_citation_reviewer": _format_call_citation_reviewer,
}


def configure_searcher_sources(source_handlers: dict[str, Callable[[str, int], list[dict]]]) -> None:
    """Configure source handlers used by call_searcher bridge (useful for tests)."""
    global _SEARCH_SOURCE_HANDLERS
    _SEARCH_SOURCE_HANDLERS = dict(source_handlers or {})


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------


def get_tool_schemas() -> list[dict]:
    """返回 OpenAI function calling 格式的 tools 列表。"""
    return _schema_get_tool_schemas()


def execute_tool(function_name: str, arguments: dict) -> str:
    """根据 function_name 路由到对应工具函数，返回字符串结果。

    未知工具或调用异常时返回错误描述字符串，不抛出异常，避免中断验证循环。
    """
    if function_name not in _TOOL_MAP:
        available = list(_TOOL_MAP.keys())
        return format_tool_error(
            tool=function_name,
            error_code="UNKNOWN_TOOL",
            message=f"Unknown tool: {function_name!r}.",
            retryable=False,
            detail={"available": available},
        )

    normalized_arguments = arguments if isinstance(arguments, dict) else {}
    try:
        return _TOOL_MAP[function_name](**normalized_arguments)
    except BaseException as exc:
        # 这里刻意扩大捕获范围：包括 KeyboardInterrupt。目标是把“中断类错误”也转换为结构化信息交给 LLM，而不是让整个主链路直接崩掉。
        if isinstance(exc, (SystemExit, GeneratorExit)):
            raise

        retryable = _is_retryable_exception(exc) or isinstance(exc, KeyboardInterrupt)
        return format_tool_error(
            tool=function_name,
            error_code="TOOL_RUNTIME_EXCEPTION",
            message=f"{function_name} raised {type(exc).__name__}: {exc}",
            retryable=retryable,
            detail={"attempt": 1, "max_attempts": 1},
        )
