import json
from pathlib import Path

import pytest

from src.agents.generator import GeneratorAgent
from src.agents.reviser import ReviserAgent
from src.agents.verifier import VerifierAgent
from src.agents.base import BaseAgent
from src.core.config import load_prompts
from src.core.orchestrator import Orchestrator
from src.memory.state import ProofState, VerificationDecision


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeLLMForGenerator:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.chat_calls = 0

    def chat(self, messages, thinking=True, stream_prefix=None):
        idx = min(self.chat_calls, len(self.outputs) - 1)
        self.chat_calls += 1
        return _Resp(self.outputs[idx])

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=20, stream_prefix=None):
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


def test_generator_agent_does_not_retry_when_contract_missing() -> None:
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
    assert resp.content == "draft without tags"
    assert llm.chat_calls == 1



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


def test_verifier_agent_leaves_llm_output_unchanged() -> None:
    class _FakeVerifierLLM:
        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            return _Resp("<verdict>CORRECT</verdict>\n<verification>ok</verification>")

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

    full_text, tool_trace, preliminary_analysis = agent.run(problem_text="demo", proof_text="<solution>x</solution>")
    # 验证 full_text 是原始 LLM 输出，未被修改
    assert full_text == "<verdict>CORRECT</verdict>\n<verification>ok</verification>"
    # 验证返回值结构
    assert tool_trace == [] or isinstance(tool_trace, list)
    assert isinstance(preliminary_analysis, str)


def test_verifier_preserves_llm_minor_flaw_verdict() -> None:
    class _FakeVerifierLLM:
        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            return _Resp(
                "<verdict>MINOR_FLAW</verdict>\n"
                "<verification>"
                "This is a minor issue: parity is a basic fact, and the informal wording does not affect the conclusion."
                "</verification>\n"
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

    full_text, tool_trace, preliminary_analysis = agent.run(
        problem_text="demo",
        proof_text="<solution>x</solution>",
    )

    # 验证 full_text 包含原始 LLM 输出的所有标签
    assert "<verdict>MINOR_FLAW</verdict>" in full_text
    assert "basic fact" in full_text
    assert "<verified_lemmas>NONE</verified_lemmas>" in full_text
    assert "<citation_review>NONE</citation_review>" in full_text


def test_verifier_handles_candidate_without_solution_tag() -> None:
    class _FakeVerifierLLM:
        def __init__(self):
            self.phase1_user_messages = []

        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                self.phase1_user_messages.append(messages[-1]["content"])
                return _Resp("phase1")
            return _Resp(
                "<verdict>MINOR_FLAW</verdict>\n"
                "<verification>Output is missing the <solution> tag and must be fixed by Reviser.</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>"
            )

        def chat_with_tools(self, messages, tools, tool_executor, max_rounds=20, stream_prefix=None, **kwargs):
            return _Resp("phase2")

        @staticmethod
        def clear_reasoning_content(messages):
            return

    llm = _FakeVerifierLLM()
    agent = VerifierAgent(
        llm_client=llm,
        prompts={
            "verifier": {
                "system": "sys",
                "phase1_user": "P1::{solution}",
                "phase2_user": "p2",
                "phase3_user": "p3",
            }
        },
        tools=[],
        tool_executor=lambda function_name, arguments: "",
    )

    full_text, tool_trace, preliminary_analysis = agent.run(
        problem_text="demo",
        proof_text="draft without xml tags",
    )

    # 验证 full_text 包含所有必要的输出标签
    assert "<verdict>MINOR_FLAW</verdict>" in full_text
    assert "<solution>" in full_text
    assert "<citation_review>NONE</citation_review>" in full_text
    # 验证 Phase 1 输入包含原始 proof_text（因为没有 <solution> 标签）
    assert llm.phase1_user_messages
    assert "draft without xml tags" in llm.phase1_user_messages[0]


def test_verifier_contract_failure_surfaces_directly() -> None:
    class _FakeVerifierLLM:
        def __init__(self):
            self.phase3_calls = 0

        def chat(self, messages, thinking=True, stream_prefix=None):
            if stream_prefix == "VERIFIER-P1":
                return _Resp("phase1")
            self.phase3_calls += 1
            # 持续返回无效格式，触发 verifier 内部降级逻辑。
            return _Resp("invalid verifier output")

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

    # 现在 verifier 不再做本地解析，直接返回 LLM 输出
    # 格式错误将由 orchestrator 在调用 parse_verification_decision 时触发
    full_text, tool_trace, preliminary_analysis = agent.run(
        problem_text="demo",
        proof_text="<solution>x</solution>",
    )
    
    # 验证 verifier 返回了 LLM 的原始输出（即使格式不正确）
    assert full_text == "invalid verifier output"
    assert isinstance(tool_trace, list)
    assert isinstance(preliminary_analysis, str)


def test_verifier_persists_verified_lemmas(tmp_path: Path) -> None:
    class _Pipeline:
        def call_generator(
            self,
            problem_text: str,
            lesson: str | None = None,
            lemma_context_items: list[str] | None = None,
        ):
            return _Resp("<solution>draft</solution>")

        def call_verifier(self, problem_text: str, proof_text: str):
            # 现在 verifier 返回3个值：verifier_response, tool_trace, preliminary_analysis
            return (
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>Lemma: if a=b then b=a. Proof: symmetry.</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>",
                [],
                "phase1",
            )

        def call_reviser(
            self,
            problem_text: str,
            previous_solution: str,
            verification_report: str,
            lemma_context_items: list[str] | None = None,
        ):
            return _Resp(previous_solution)

        def call_final(self, problem_text: str, current_solution: str, last_verifier_decision: str, last_verification_report: str):
            return "PROGRESS", "", current_solution, ""

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
    assert out.status == "SUCCESS"

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


def test_reviser_agent_does_not_retry_when_solution_missing() -> None:
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
    assert resp.content == "draft only"
    assert llm.chat_calls == 1


def test_base_agent_clears_stage_memory_after_each_run() -> None:
    class _RecordingLLM:
        def __init__(self):
            self.inputs = []

        def chat(self, messages, thinking=True, stream_prefix=None):
            # 璁板綍姣忔杩涘叆妯″瀷鏃剁殑杈撳叆娑堟伅锛岀‘淇濇病鏈夎法 run 姹℃煋銆?
            self.inputs.append([dict(item) for item in messages])
            return _Resp("ok")

    llm = _RecordingLLM()
    agent = BaseAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    agent.run("first")
    agent.run("second")

    assert len(llm.inputs) == 2
    assert len(llm.inputs[0]) == 2
    assert len(llm.inputs[1]) == 2
    assert llm.inputs[0][1]["content"] == "first"
    assert llm.inputs[1][1]["content"] == "second"
    assert agent.messages == []


def test_reviser_reasoning_not_persisted_in_history(tmp_path: Path) -> None:
    class _Pipeline:
        def __init__(self):
            self.verifier_calls = 0

        def call_generator(
            self,
            problem_text: str,
            lesson: str | None = None,
            lemma_context_items: list[str] | None = None,
        ):
            return _Resp("<solution>draft</solution>", reasoning_content="gen-think")

        def call_verifier(self, problem_text: str, proof_text: str):
            self.verifier_calls += 1
            if self.verifier_calls == 1:
                return (
                    "<verdict>MINOR_FLAW</verdict>\n"
                    "<verification>need revise</verification>\n"
                    "<verified_lemmas>NONE</verified_lemmas>\n"
                    "<citation_review>NONE</citation_review>",
                    VerificationDecision.MINOR_FLAW,
                    "need revise",
                    [],
                    "phase1",
                )
            return (
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>",
                VerificationDecision.CORRECT,
                "",
                [],
                "phase1",
            )

        def call_reviser(
            self,
            problem_text: str,
            previous_solution: str,
            verification_report: str,
            lemma_context_items: list[str] | None = None,
        ):
            return _Resp("<solution>revised</solution>", reasoning_content="rev-think-should-not-log")

        def call_final(self, problem_text: str, current_solution: str, last_verifier_decision: str, last_verification_report: str):
            return "PROGRESS", "", current_solution, ""

    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=2,
        pipeline=_Pipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-reviser-no-reason", problem_text="demo")

    out = orchestrator.run(state)
    assert out.status == "SUCCESS"

    history_path = runs_root / "p-reviser-no-reason" / "history.jsonl"
    lines = [line.strip() for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]

    reviser_events = [item for item in events if item.get("node") == "REVISER"]
    assert len(reviser_events) == 1
    assert "reasoning_content" not in reviser_events[0]

    generator_events = [item for item in events if item.get("node") == "GENERATOR"]
    assert len(generator_events) >= 1
    assert "reasoning_content" in generator_events[0]

