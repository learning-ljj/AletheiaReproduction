import json
from pathlib import Path

from src.agents.generator import GeneratorAgent
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.tools.registry import configure_searcher_sources, execute_tool, get_tool_schemas


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = ""


class _FakeLLMGeneratorWithSearcher:
    def __init__(self):
        self.last_tool_payload = None

    def chat(self, messages, thinking=True, stream_prefix=None):
        return _Resp("<verdict>PARTIAL</verdict>\n<solution>fallback</solution>")

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=10, stream_prefix=None):
        self.last_tool_payload = tool_executor("call_searcher", {"query": "compactness theorem"})
        return _Resp(
            "<verdict>PARTIAL</verdict>\n"
            "<solution>Used retrieved note [cite:runs/p-int/artifact/papers/title_dummy.md]</solution>"
        )



def test_generator_searcher_bridge_integration(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    memory = ProblemMemory(problem_id="p-int", runs_root=runs_root)
    set_current_problem_memory(memory)

    def _fake_source(query: str, limit: int) -> list[dict]:
        return [
            {
                "title": "dummy",
                "abstract": "retrieved abstract",
                "authors": ["A"],
                "url": "https://example.org/dummy",
            }
        ]

    configure_searcher_sources({"fake": _fake_source})

    llm = _FakeLLMGeneratorWithSearcher()
    agent = GeneratorAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=get_tool_schemas(),
        tool_executor=execute_tool,
        max_tool_rounds=2,
    )

    resp = agent.run(problem_text="demo problem")
    assert "<solution>" in resp.content

    tool_payload = json.loads(llm.last_tool_payload)
    assert tool_payload["status"] == "OK"
    assert tool_payload["paper_count"] == 1
    assert len(tool_payload["papers"]) == 1

    paper_files = list(memory.papers_dir.glob("*.md"))
    assert len(paper_files) == 1

    # cleanup current context for other tests
    set_current_problem_memory(None)
    configure_searcher_sources({})
