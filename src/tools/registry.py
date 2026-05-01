"""工具注册表：OpenAI Function Calling schema 与统一执行分发。"""

from typing import Callable

# 还没把 Search 功能改为 Agent，现在是 pipeline , 在 tools/search_papers/
from src.tools.search_papers.searcher import SearchPipelne
from src.tools.review_citation import review_citation

from src.memory.problem_memory import get_current_problem_memory
from src.tools.artifact_reader import read_artifact
from src.tools.code_executor import run_python
from src.tools.format import (
    format_tool_error,
    format_tool_success,
)


class ToolExecutor:
    """按工具名执行统一包络的工具调用器。"""

    def __init__(self, source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None):
        self.source_handlers = dict(source_handlers or {})

    def __call__(self, function_name: str, arguments: dict) -> str:
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
            if function_name == "call_searcher":
                return _format_call_searcher(
                    source_handlers=self.source_handlers,
                    **normalized_arguments,
                )
            return _TOOL_MAP[function_name](**normalized_arguments)
        except BaseException as exc:
            # 这里刻意扩大捕获范围：包括 KeyboardInterrupt。目标是把“中断类错误”也转换为结构化信息交给 LLM，而不是让整个主链路直接崩掉。
            if isinstance(exc, (SystemExit, GeneratorExit)):
                raise

            retryable = isinstance(exc, (TimeoutError, ConnectionError, OSError)) or isinstance(exc, KeyboardInterrupt)
            return format_tool_error(
                tool=function_name,
                error_code="TOOL_RUNTIME_EXCEPTION",
                message=f"{function_name} raised {type(exc).__name__}: {exc}",
                retryable=retryable,
                detail={"attempt": 1, "max_attempts": 1},
            )


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
    *,
    source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None,
    query: str | None = None,
    query_bundle: list[str] | None = None,
    **extra_args,
) -> str:
    """调用 SearchPipelne 并返回检索摘要与落盘路径。"""
    problem_memory = get_current_problem_memory()
    if problem_memory is None:
        return format_tool_error(
            tool="call_searcher",
            error_code="NO_PROBLEM_MEMORY",
            message="ProblemMemory context is missing for call_searcher.",
            retryable=False,
        )
    if not source_handlers:
        return format_tool_error(
            tool="call_searcher",
            error_code="NO_SEARCH_SOURCES_CONFIGURED",
            message="Searcher source handlers are not configured.",
            retryable=False,
        )

    search_pipeline = SearchPipelne(
        problem_memory=problem_memory,
        source_handlers=source_handlers,
    )

    # 重点说明：search_pipeline.run 不仅会返回 papers，也会返回 errors/recovered_errors。
    # 这些错误字段会原样回传给 LLM，帮助模型决定下一步动作，而不是像传统流程那样“空结果即静默降级”。
    result = search_pipeline.run(query=query, query_bundle=query_bundle)
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


def _format_read_artifact(path: str, layer: int) -> str:
    """按层读取 artifact 文档（artifact_reader 内部已统一包络）。"""
    return read_artifact(path=path, layer=layer)


def _format_review_citation(
    cites: list[str] | None = None,
    **extra_args,
) -> str:
    """调用引用内容检查工具，并返回统一包络。"""
    problem_memory = get_current_problem_memory()
    if problem_memory is None:
        return format_tool_error(
            tool="review_citation",
            error_code="NO_PROBLEM_MEMORY",
            message="ProblemMemory context is missing for review_citation.",
            retryable=False,
        )

    review = review_citation(problem_memory=problem_memory, cites=cites or [])
    payload: dict = dict(review)
    if extra_args:
        payload["extra_args"] = extra_args
    return format_tool_success(tool="review_citation", data=payload)


# 函数名 → 可调用对象的映射
_TOOL_MAP: dict = {
    "run_python": _format_run_python,
    "call_searcher": _format_call_searcher,
    "read_artifact": _format_read_artifact,
    "review_citation": _format_review_citation,
}
