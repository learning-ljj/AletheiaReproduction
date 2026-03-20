"""工具注册表：OpenAI Function Calling schema 与统一执行分发。"""

import json

from src.tools.code_executor import run_python
from src.tools.web_search import read_arxiv_latex, search_arxiv
from src.tools.wiki_search import search_wikipedia

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
            "name": "search_wikipedia",
            "description": (
                "Search Wikipedia for a concept, theorem, or definition. "
                "Use this FIRST for general mathematical concepts, named theorems, or "
                "well-known results before trying search_arxiv. "
                "Use precise queries (single theorem name or concept), not keyword soup. "
                "If the first result is off-topic, reformulate once with a clearer term. "
                "Returns the cleaned page content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. theorem name, concept).",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search for academic papers on arXiv by query. "
                "Use this ONLY for specific academic paper citations or when "
                "search_wikipedia does not provide sufficient information. "
                "For general named theorems or well-known results, prefer search_wikipedia first. "
                "Use concrete paper-oriented queries (title fragment / author / exact topic) rather than "
                "broad keyword concatenation. "
                "Returns a list of papers with arXiv ID, title, authors, published date, "
                "and abstract snippet. If no results are found, flag the citation as a "
                "potential hallucination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. paper title, theorem name, author).",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_arxiv_latex",
            "description": (
                "Download an arXiv paper's LaTeX source and extract its abstract and key "
                "sections (Main Results / Theorems / Key Findings / Conclusion). "
                "Use this AFTER search_arxiv confirms a paper exists, to verify theorem "
                "statements, preconditions, and correct usage in the solution. "
                "Returns up to 6,000 characters of the extracted key sections; "
                "if no structured sections are found, returns the first 6,000 characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "The arXiv ID of the paper (e.g. '2501.12345v1').",
                    }
                },
                "required": ["arxiv_id"],
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


def _format_search_wikipedia(query: str) -> str:
    """执行 Wikipedia 搜索并返回结果字符串。"""
    return search_wikipedia(query)


def _format_search_arxiv(query: str) -> str:
    """执行 arXiv 搜索并序列化结果为 JSON 字符串。"""
    return json.dumps(search_arxiv(query), ensure_ascii=False)


_ARXIV_LATEX_MAX_CHARS = 6_000  # 限制 LaTeX 关键章节返回量（摘要+主要结果），避免填满 LLM 上下文窗口
# 说明：read_arxiv_latex 现在调用 _extract_key_sections 提取摘要和关键章节，
# 6000 字符已足够包含 abstract + 一两个主要定理/结果段落。


def _format_read_arxiv_latex(arxiv_id: str) -> str:
    """下载并返回截断后的 LaTeX 源码（限制 30,000 字符）。"""
    return read_arxiv_latex(arxiv_id, max_chars=_ARXIV_LATEX_MAX_CHARS)


# 函数名 → 可调用对象的映射
_TOOL_MAP: dict = {
    "run_python": _format_run_python,
    "search_wikipedia": _format_search_wikipedia,
    "search_arxiv": _format_search_arxiv,
    "read_arxiv_latex": _format_read_arxiv_latex,
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
