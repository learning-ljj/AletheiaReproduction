import json
from pathlib import Path

from src.agents.citation_reviewer import CitationReviewerAgent
from src.agents.verifier import VerifierAgent
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.tools.registry import ToolExecutor
from src.tools.schemas import get_tool_schemas
from src.utils.parsing.parser import extract_xml_tag



def test_citation_reviewer_detects_missing_path(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-cite-missing", runs_root=tmp_path / "runs")
    reviewer = CitationReviewerAgent(problem_memory=memory)

    review = reviewer.review(cites=["artifact/papers/missing.md"], claim_spans=[])
    assert review["fail_count"] == 1
    assert review["items"][0]["reason"] == "PATH_NOT_FOUND"



def test_citation_reviewer_passes_existing_path(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-cite-ok", runs_root=tmp_path / "runs")
    cite_file = memory.add_paper("demo paper", filename="demo.md")
    assert cite_file.exists()

    reviewer = CitationReviewerAgent(problem_memory=memory)
    review = reviewer.review(cites=["artifact/papers/demo.md"], claim_spans=[])

    assert review["fail_count"] == 0
    assert review["items"][0]["passed"] is True



def test_verifier_triggers_citation_review_on_demand(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-verifier-cite", runs_root=tmp_path / "runs")
    memory.add_paper("claim", filename="demo.md")
    set_current_problem_memory(memory)

    class _FakeVerifierLLM:
        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            return _Resp(
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>{\"fail_count\": 0, \"items\": [{\"passed\": true}]}</citation_review>"
            )

        def chat_with_tools(self, messages, tools, tool_executor, max_rounds=20, stream_prefix=None, **kwargs):
            return _Resp("phase2")

        @staticmethod
        def clear_reasoning_content(messages):
            return

    agent = VerifierAgent(
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
    verifier_response, tool_trace, preliminary_analysis = agent.run(problem_text="demo", proof_text=proof_text)

    review_text = extract_xml_tag(verifier_response, "citation_review")
    review_payload = json.loads(review_text)
    assert review_payload["fail_count"] == 0
    assert review_payload["items"][0]["passed"] is True

    set_current_problem_memory(None)


def test_verifier_ignores_citations_outside_solution_block() -> None:
    class _FakeVerifierLLM:
        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            return _Resp(
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>"
            )

        def chat_with_tools(self, messages, tools, tool_executor, max_rounds=20, stream_prefix=None, **kwargs):
            return _Resp("phase2")

        @staticmethod
        def clear_reasoning_content(messages):
            return

    agent = VerifierAgent(
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

    proof_text = (
        "<thinking>Template note [cite:placeholder/path.md]</thinking>\n"
        "<solution>No citation is used in the final proof.</solution>"
    )
    verifier_response, tool_trace, preliminary_analysis = agent.run(problem_text="demo", proof_text=proof_text)

    review_text = extract_xml_tag(verifier_response, "citation_review")
    assert review_text == "NONE"


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeLLMVerifierWithCitationTool:
    def __init__(self):
        self.last_tool_payload: str | None = None

    def chat(self, messages, thinking=True, stream_prefix=None):
        if stream_prefix == "VERIFIER-P1":
            return _Resp("phase1")
        return _Resp(
            "<verdict>CORRECT</verdict>\n"
            "<verification>ok</verification>\n"
            "<verified_lemmas>NONE</verified_lemmas>\n"
            "<citation_review>{\"fail_count\": 0, \"items\": [{\"passed\": true}]}</citation_review>"
        )

    def chat_with_tools(self, messages, tools, tool_executor, max_rounds=20, stream_prefix=None, **kwargs):
        self.last_tool_payload = tool_executor(
            "call_citation_reviewer",
            {
                "cites": ["artifact/papers/demo.md"],
                "claim_spans": ["claim"],
            },
        )
        return _Resp("phase2")

    @staticmethod
    def clear_reasoning_content(messages):
        return


def test_verifier_citation_review_requires_llm_contract(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-verifier-tool-cite", runs_root=tmp_path / "runs")
    memory.add_paper("claim", filename="demo.md")
    set_current_problem_memory(memory)

    llm = _FakeLLMVerifierWithCitationTool()
    agent = VerifierAgent(
        llm_client=llm,
        prompts={
            "verifier": {
                "system": "sys",
                "phase1_user": "p1",
                "phase2_user": "p2",
                "phase3_user": "p3",
            }
        },
        tools=get_tool_schemas(),
        tool_executor=ToolExecutor(),
    )

    proof_text = "<solution>claim [cite:artifact/papers/demo.md]</solution>"
    full_text, _, _, _, _ = agent.run(problem_text="demo", proof_text=proof_text)

    review_payload = json.loads(extract_xml_tag(full_text, "citation_review"))
    assert llm.last_tool_payload is not None
    assert review_payload["fail_count"] == 0
    assert review_payload["items"][0]["passed"] is True

    set_current_problem_memory(None)
