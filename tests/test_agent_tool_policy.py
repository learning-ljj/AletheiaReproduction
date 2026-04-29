import json

from src.core.agent import (
    _AGENT_TOOL_ALLOWLIST,
    AgentPipeline,
    _build_scoped_tool_executor,
    _filter_tool_schemas,
)
from src.tools.registry import execute_tool


class _DummyLLM:
    pass


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "demo",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_filter_tool_schemas_by_allowlist() -> None:
    schemas = [
        _schema("run_python"),
        _schema("call_searcher"),
        _schema("read_artifact_layer"),
        _schema("call_citation_reviewer"),
    ]
    filtered = _filter_tool_schemas(schemas, {"run_python", "read_artifact_layer"})

    assert [item["function"]["name"] for item in filtered] == ["run_python", "read_artifact_layer"]


def test_scoped_tool_executor_blocks_disallowed_calls() -> None:
    calls: list[tuple[str, dict]] = []

    def _base(function_name: str, arguments: dict) -> str:
        calls.append((function_name, arguments))
        return "ok"

    scoped = _build_scoped_tool_executor(_base, {"run_python"})
    blocked = scoped("call_searcher", {"query": "demo"})
    allowed = scoped("run_python", {"code": "print(1)"})

    assert "not permitted" in blocked
    assert allowed == "ok"
    assert calls == [("run_python", {"code": "print(1)"})]


def test_agent_runtime_uses_stage_tool_allowlists() -> None:
    schemas = [
        _schema("run_python"),
        _schema("call_searcher"),
        _schema("read_artifact_layer"),
        _schema("call_citation_reviewer"),
    ]
    prompts = {
        "generator": {"system": "gen"},
        "reviser": {"system": "rev"},
        "verifier": {
            "system": "ver",
            "phase1_user": "p1",
            "phase2_user": "p2",
            "phase3_user": "p3",
        },
    }

    runtime = AgentPipeline(
        llm_client=_DummyLLM(),
        prompts=prompts,
        tool_schemas=schemas,
        tool_executor=lambda function_name, arguments: "ok",
    )

    generator_names = {item["function"]["name"] for item in runtime.generator_agent.tools}
    reviser_names = {item["function"]["name"] for item in runtime.reviser_agent.tools}
    verifier_names = {item["function"]["name"] for item in runtime.verifier_agent.tools}

    assert generator_names == _AGENT_TOOL_ALLOWLIST["generator"]
    assert reviser_names == _AGENT_TOOL_ALLOWLIST["reviser"]
    assert verifier_names == _AGENT_TOOL_ALLOWLIST["verifier"]


def test_execute_tool_unknown_returns_structured_error() -> None:
    payload = json.loads(execute_tool("__unknown_tool__", {}))
    assert payload["status"] == "ERROR"
    assert payload["error"]["error_code"] == "UNKNOWN_TOOL"
    assert payload["error"]["retryable"] is False
