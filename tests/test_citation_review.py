import json
from pathlib import Path

from src.agents.citation_reviewer import CitationReviewerAgent
from src.agents.verifier import VerifierAgent
from src.core.state import VerificationDecision
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.utils.parser import extract_xml_tag



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
    memory.add_paper("content", filename="demo.md")
    set_current_problem_memory(memory)

    def _runner(llm, prompts, problem_text, proof_text, tools, tool_executor):
        return (
            "<verdict>CORRECT</verdict>\n"
            "<verification>ok</verification>\n"
            "<verified_lemmas>NONE</verified_lemmas>",
            VerificationDecision.CORRECT,
            "",
            [],
            "phase1",
        )

    agent = VerifierAgent(
        llm_client=object(),
        prompts={},
        tools=[],
        tool_executor=lambda function_name, arguments: "",
        verifier_runner=_runner,
    )

    proof_text = "<solution>claim [cite:artifact/papers/demo.md]</solution>"
    full_text, _, _, _, _ = agent.run(problem_text="demo", proof_text=proof_text)

    review_text = extract_xml_tag(full_text, "citation_review")
    review_payload = json.loads(review_text)
    assert review_payload["fail_count"] == 0
    assert review_payload["items"][0]["passed"] is True

    set_current_problem_memory(None)
