"""Tool schema registry for OpenAI function calling."""

# 这里定义的是“给模型看的工具说明书”，不是工具实现本身。
# 模型会根据 name/description/parameters 生成 tool_calls，
# 真实执行由 src/tools/registry.py 的 execute_tool 完成。
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
                "Bridge call to SearcherAgent full retrieval chain. "
                "Use this whenever external knowledge retrieval is required. "
                "Searcher executes query expansion, multi-source retrieval, dedup, candidate-claim extraction, "
                "and persists layered markdown artifacts under runs/{problem_id}/artifact/papers."
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
                "required": ["query", "query_bundle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
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
    {
        "type": "function",
        "function": {
            "name": "call_citation_reviewer",
            "description": (
                "Trigger CitationReviewerAgent to validate cite path existence and claim-source semantic support. "
                "Use this in verifier stage when [cite:path] markers are detected in the candidate solution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Citation paths extracted from [cite:path] markers.",
                    },
                    "claim_spans": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sentence-level claim spans aligned with cites by index.",
                    },
                },
                "required": ["cites"],
            },
        },
    },
]


def get_tool_schemas() -> list[dict]:
    """Return tool schema list in OpenAI function-calling format."""
    # 直接返回静态 schema 列表，调用侧不要原地修改该对象。
    return _TOOL_SCHEMAS
