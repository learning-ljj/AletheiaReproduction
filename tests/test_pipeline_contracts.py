from pathlib import Path

from src.agents.generator import GeneratorAgent
from src.agents.reviser import ReviserAgent
from src.agents.verifier import VerifierAgent
from src.core.config import load_prompts
from src.core.orchestrator import Orchestrator
from src.core.state import ProofState, RunStatus, VerificationDecision


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = ""


class _FakeLLMForGenerator:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.chat_calls = 0

    def chat(self, messages, thinking=True, stream_prefix=None):
        idx = min(self.chat_calls, len(self.outputs) - 1)
        self.chat_calls += 1
        return _Resp(self.outputs[idx])

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=10, stream_prefix=None):
        idx = min(self.chat_calls, len(self.outputs) - 1)
        self.chat_calls += 1
        return _Resp(self.outputs[idx])


class _NoopLogger:
    def append_raw_event(self, problem_id: str, payload: dict) -> None:
        return

    def save_final_output_markdown(self, problem_id: str, final_output: str) -> None:
        return


class _SimpleFinalizer:
    def build_final_output(
        self,
        success: bool,
        solution_text: str | None,
        failure_reason: str | None,
        *,
        partial: bool = False,
        assessment_output: str | None = None,
        preserve_xml: bool = False,
        references: list[str] | None = None,
        warning_summary: str | None = None,
    ) -> str:
        return "ok"


def test_prompt_generator_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["generator"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "<lemma>" in text
    assert "</lemma>" in text
    assert "[cite:" in text


def test_generator_agent_returns_structured_output() -> None:
    llm = _FakeLLMForGenerator([
        "<verdict>PARTIAL</verdict>\n<solution><lemma>L1</lemma>step</solution>",
    ])
    agent = GeneratorAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(problem_text="demo problem", error_lessons="avoid gap")
    assert "<verdict>" in resp.content
    assert "<solution>" in resp.content


def test_generator_agent_retries_once_when_contract_missing() -> None:
    llm = _FakeLLMForGenerator([
        "draft without tags",
        "<verdict>PARTIAL</verdict>\n<solution>fixed</solution>",
    ])
    agent = GeneratorAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(problem_text="demo")
    assert "<solution>" in resp.content
    assert llm.chat_calls == 2



def test_prompt_verifier_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["verifier"]["phase3_user"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<verification>" in text
    assert "</verification>" in text
    assert "<verified_lemmas>" in text
    assert "</verified_lemmas>" in text
    assert "<citation_review>" in text
    assert "</citation_review>" in text


def test_verifier_agent_adds_optional_contract_blocks() -> None:
    def _runner(llm, prompts, problem_text, proof_text, tools, tool_executor):
        return (
            "<verdict>CORRECT</verdict>\n<verification>ok</verification>",
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

    full_text, decision, _, _, _ = agent.run(problem_text="demo", proof_text="<solution>x</solution>")
    assert decision == VerificationDecision.CORRECT
    assert "<verified_lemmas>" in full_text
    assert "<citation_review>" in full_text


def test_verifier_persists_verified_lemmas(tmp_path: Path) -> None:
    class _Pipeline:
        def call_generator(self, problem_text: str, lesson: str | None = None):
            return _Resp("<solution>draft</solution>")

        def call_verifier(self, problem_text: str, proof_text: str):
            return (
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>Lemma: if a=b then b=a. Proof: symmetry.</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>",
                VerificationDecision.CORRECT,
                "",
                [],
                "phase1",
            )

        def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
            return _Resp(previous_solution)

        def call_final(self, problem_text: str, current_solution: str, last_verifier_decision: str, last_verification_report: str):
            return RunStatus.PARTIAL.value, "", current_solution, ""

    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=1,
        pipeline=_Pipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-verifier", problem_text="demo")

    out = orchestrator.run(state)
    assert out.status == RunStatus.SUCCESS

    lemma_file = runs_root / "p-verifier" / "artifact" / "lemmas" / "001.md"
    assert lemma_file.exists()
    assert "symmetry" in lemma_file.read_text(encoding="utf-8")



def test_prompt_reviser_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["reviser"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "[cite:" in text


def test_reviser_agent_returns_solution_block() -> None:
    llm = _FakeLLMForGenerator([
        "<verdict>PARTIAL</verdict>\n<solution>patched solution</solution>",
    ])
    agent = ReviserAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(
        problem_text="demo",
        previous_solution="<solution>old</solution>",
        verification_report="fix Step 2",
    )
    assert "<solution>" in resp.content


def test_reviser_agent_retries_when_solution_missing() -> None:
    llm = _FakeLLMForGenerator([
        "draft only",
        "<verdict>PARTIAL</verdict>\n<solution>fixed reviser output</solution>",
    ])
    agent = ReviserAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(
        problem_text="demo",
        previous_solution="<solution>old</solution>",
        verification_report="fix Step 2",
    )
    assert "<solution>" in resp.content
    assert llm.chat_calls == 2
