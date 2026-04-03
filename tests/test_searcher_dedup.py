from pathlib import Path

from src.agents.base import BaseAgent
from src.agents.searcher import SearcherAgent
from src.memory.problem_memory import ProblemMemory
from src.tools.search_sources import build_default_source_handlers


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeLLMClient:
    def __init__(self):
        self.chat_calls = 0
        self.chat_with_tools_calls = 0
        self.last_max_tool_rounds = None
        self.chat_message_snapshots = []
        self.chat_with_tools_message_snapshots = []

    def chat(self, messages, thinking=True, stream_prefix=None):
        self.chat_calls += 1
        self.chat_message_snapshots.append([dict(m) for m in messages])
        return _Resp(content="plain-result", reasoning_content="plain-reason")

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=10, stream_prefix=None):
        self.chat_with_tools_calls += 1
        self.last_max_tool_rounds = max_tool_rounds
        self.chat_with_tools_message_snapshots.append([dict(m) for m in messages])
        tool_executor("dummy", {})
        return _Resp(content="tool-result", reasoning_content="tool-reason")



def test_base_agent_stage_memory_reset_between_runs() -> None:
    llm = _FakeLLMClient()
    agent = BaseAgent(llm_client=llm, system_prompt="sys", max_tool_rounds=5)

    first_output = agent.run("payload-one")
    assert first_output.content == "plain-result"
    assert first_output.reasoning_content == "plain-reason"
    assert agent.messages == []
    first_snapshot = llm.chat_message_snapshots[0]
    assert any("payload-one" in (m.get("content") or "") for m in first_snapshot)

    second_output = agent.run("payload-two")
    assert second_output.content == "plain-result"
    assert second_output.reasoning_content == "plain-reason"
    assert agent.messages == []

    second_snapshot = llm.chat_message_snapshots[1]
    assert any("payload-two" in (m.get("content") or "") for m in second_snapshot)
    assert not any("payload-one" in (m.get("content") or "") for m in second_snapshot)



def test_base_agent_respects_max_tool_rounds() -> None:
    llm = _FakeLLMClient()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dummy",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent = BaseAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=tools,
        tool_executor=lambda function_name, arguments: "ok",
        max_tool_rounds=3,
    )

    output = agent.run({"problem": "demo"})
    assert output.content == "tool-result"
    assert output.reasoning_content == "tool-reason"
    assert llm.chat_with_tools_calls == 1
    assert llm.last_max_tool_rounds == 3



def test_base_agent_returns_final_text_without_tools() -> None:
    llm = _FakeLLMClient()
    agent = BaseAgent(llm_client=llm, system_prompt="sys")

    output = agent.run("only-text")
    assert output.content == "plain-result"
    assert output.reasoning_content == "plain-reason"
    assert llm.chat_calls == 1
    assert llm.chat_with_tools_calls == 0


def test_searcher_dedup_persists_unique_papers(tmp_path: Path) -> None:
    def _source_a(query: str, limit: int) -> list[dict]:
        return [
            {
                "title": "Paper with DOI",
                "doi": "10.1000/demo-doi",
                "abstract": "A",
                "authors": ["Alice"],
                "url": "https://example.org/doi",
            },
            {
                "title": "Paper with DOI duplicate",
                "doi": "10.1000/demo-doi",
                "abstract": "A2",
                "authors": ["Alice"],
                "url": "https://example.org/doi-dup",
            },
            {
                "title": "Paper with arXiv",
                "arxiv_id": "2501.12345",
                "abstract": "B",
                "authors": ["Bob"],
                "url": "https://arxiv.org/abs/2501.12345",
            },
        ]

    def _source_b(query: str, limit: int) -> list[dict]:
        return [
            {
                "title": "Paper with arXiv duplicate",
                "arxiv_id": "2501.12345",
                "abstract": "B2",
                "authors": ["Bob"],
                "url": "https://arxiv.org/abs/2501.12345v2",
            },
            {
                "title": "Title only unique",
                "abstract": "C",
                "authors": ["Carol"],
                "url": "https://example.org/title-only",
            },
            {
                "title": "Title only unique",
                "abstract": "C2",
                "authors": ["Carol"],
                "url": "https://example.org/title-only-dup",
            },
        ]

    memory = ProblemMemory(problem_id="p-search", runs_root=tmp_path / "runs")
    agent = SearcherAgent(
        problem_memory=memory,
        source_handlers={"source_a": _source_a, "source_b": _source_b},
        limit_per_query=5,
    )

    result = agent.run(query="graph theory")
    assert result["count"] == 3

    paper_files = sorted(memory.papers_dir.glob("*.md"))
    assert len(paper_files) == 3

    second_result = agent.run(query="graph theory")
    assert second_result["count"] == 3
    assert len(list(memory.papers_dir.glob("*.md"))) == 3


def test_default_search_source_handlers_are_non_empty() -> None:
    handlers = build_default_source_handlers({})
    assert "openalex" in handlers
    assert "arxiv" in handlers
