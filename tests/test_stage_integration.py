import json
from pathlib import Path

from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.tools.registry import configure_searcher_sources, execute_tool, get_tool_schemas
from src.utils.parsing.parser import extract_xml_tag


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = ""


class _FakeLLMGeneratorWithSearcher:
    def __init__(self):
        self.last_tool_payload = None

    def chat(self, messages, thinking=True, stream_prefix=None):
        return _Resp("<verdict>PARTIAL</verdict>\n<solution>fallback</solution>")

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=20, stream_prefix=None):
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
    assert tool_payload["data"]["paper_count"] == 1
    assert len(tool_payload["data"]["papers"]) == 1

    paper_files = list(memory.papers_dir.glob("*.md"))
    assert len(paper_files) == 1

    # cleanup current context for other tests
    set_current_problem_memory(None)
    configure_searcher_sources({})


def test_verifier_citation_reviewer_stage_integration(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-vcite", runs_root=tmp_path / "runs")
    memory.add_paper("claim", filename="demo.md")
    set_current_problem_memory(memory)

    class _FakeVerifierLLM:
        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            return _Resp(
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>"
            )

        def chat_with_tools(self, messages, tools, tool_executor, stream_prefix=None, max_tool_rounds=20):
            return _Resp("phase2")

        @staticmethod
        def clear_reasoning_content(messages):
            return

    verifier = VerifierAgent(
        llm_client=_FakeVerifierLLM(),
        prompts={
            "verifier": {
                "system": "sys",
                "phase1_user": "p1",
                "phase2_user": "p2",
                "phase3_user": "p3",
            }
        },
        tools=[],
        tool_executor=lambda function_name, arguments: "",
    )

    proof_text = "<solution>claim [cite:artifact/papers/demo.md]</solution>"
    full_text, _, _, _, _ = verifier.run(problem_text="demo", proof_text=proof_text)

    review_payload = json.loads(extract_xml_tag(full_text, "citation_review"))
    assert review_payload["fail_count"] == 0
    assert review_payload["items"][0]["passed"] is True

    set_current_problem_memory(None)


def test_call_searcher_requires_configured_sources(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-no-source", runs_root=tmp_path / "runs")
    set_current_problem_memory(memory)
    configure_searcher_sources({})

    payload = json.loads(execute_tool("call_searcher", {"query": "demo"}))
    assert payload["status"] == "ERROR"
    assert payload["error"]["error_code"] == "NO_SEARCH_SOURCES_CONFIGURED"

    set_current_problem_memory(None)
