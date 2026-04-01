"""工具注册表：OpenAI Function Calling schema 与统一执行分发。"""

import json

from src.tools.artifact_reader import read_artifact_layer
from src.tools.code_executor import run_python

# ------------------------------------------------------------------
# OpenAI function calling 兼容的 tools schema
# ------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return stdout/stderr. Use this to verify arithmetic, algebraic, or numerical steps. "
                "Before writing code, verify API availability in standard library modules; do NOT call non-existent functions (e.g., `math.phi`). "
                "If Euler's totient is needed, implement a local `phi(n)` helper in the snippet. "
                "Requirements for checks involving fractions or rational expressions:\n"
                "- For formulas containing fractions or rational expressions, do NOT perform comparisons by converting the theoretical expression into integer division using `//`.\n"
                "- Prefer exact arithmetic using `fractions.Fraction` or compare by cross-multiplication to ensure precise equality checks.\n"
                "- If rounding or floor operations are intentionally used (e.g., `//` or `math.floor`), explicitly state in the output that this is part of the problem definition and not an implementation approximation.\n"
                "Code snippets must be self-contained and not rely on prior execution state; always print labeled final checked values for reproducibility. "
                "For script-like checks, include a short PASS/FAIL summary line. Avoid OOM or exponential-time brute-force."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_searcher",
            "description": (
                "Bridge call to the SearcherAgent retrieval chain. "
                "Use this whenever external knowledge retrieval is required. "
                "Current phase provides a placeholder response only; the full chain "
                "will be wired in later tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Primary retrieval query.",
                    },
                    "query_bundle": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional expanded retrieval queries.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact_layer",
            "description": (
                "Read one specific layer from an artifact markdown file under runs/{problem_id}/artifact. "
                "layer=1 reads YAML frontmatter summary, layer=2 reads Layer2 body, "
                "layer=3 reads Layer3 source metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Artifact markdown file path under runs/{problem_id}/artifact.",
                    },
                    "layer": {
                        "type": "integer",
                        "description": "Target layer index: 1, 2, or 3.",
                    },
                },
                "required": ["path", "layer"],
            },
        },
    },
]

def _format_run_python(code: str) -> str:
    """执行代码并格式化返回结果为字符串。"""
    result = run_python(code)
    parts = []
    if result["stdout"]:
        parts.append(f"stdout:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"stderr:\n{result['stderr']}")
    if not parts:
        parts.append("(no output)")
    parts.append(f"exit_code: {result['exit_code']}")
    return "\n".join(parts)


def _format_call_searcher(
    query: str | None = None,
    query_bundle: list[str] | None = None,
    **extra_args,
) -> str:
    """SearcherAgent 桥接占位实现（A10阶段）。"""
    payload = {
        "status": "NOT_IMPLEMENTED",
        "tool": "call_searcher",
        "message": "SearcherAgent bridge is not wired yet. It will be implemented in Phase D.",
        "query": query,
        "query_bundle": query_bundle or [],
    }
    if extra_args:
        payload["extra_args"] = extra_args
    return json.dumps(payload, ensure_ascii=False)


def _format_read_artifact_layer(path: str, layer: int) -> str:
    """按层读取 artifact 文档。"""
    return read_artifact_layer(path=path, layer=layer)


# 函数名 → 可调用对象的映射
_TOOL_MAP: dict = {
    "run_python": _format_run_python,
    "call_searcher": _format_call_searcher,
    "read_artifact_layer": _format_read_artifact_layer,
}


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------


def get_tool_schemas() -> list[dict]:
    """返回 OpenAI function calling 格式的 tools 列表。"""
    return _TOOL_SCHEMAS


def execute_tool(function_name: str, arguments: dict) -> str:
    """根据 function_name 路由到对应工具函数，返回字符串结果。

    未知工具或调用异常时返回错误描述字符串，不抛出异常，避免中断验证循环。
    """
    if function_name not in _TOOL_MAP:
        available = list(_TOOL_MAP.keys())
        return f"[TOOL ERROR] Unknown tool: {function_name!r}. Available: {available}"
    try:
        return _TOOL_MAP[function_name](**arguments)
    except Exception as exc:
        return f"[TOOL ERROR] {function_name} raised {type(exc).__name__}: {exc}"
