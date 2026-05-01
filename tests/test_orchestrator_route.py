import json
from pathlib import Path

import pytest

from src.core.orchestrator import Orchestrator
from src.memory.state import ProofState, VerificationDecision


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _NoopLogger:
    def append_raw_event(self, problem_id: str, payload: dict) -> None:
        # A13 涔嬪悗锛宺aw event 鐢?ProblemMemory 璐熻矗锛岃繖閲屼繚鎸佸吋瀹瑰崰浣嶃€?
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
        status = "success" if success else ("partial" if partial else "failed")
        return (
            f"status={status};reason={failure_reason};solution={solution_text};"
            f"assessment={assessment_output};references={references};warning={warning_summary}"
        )


class _SuccessPipeline:
    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        return _Resp(content="<solution>ok</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>CORRECT</verdict>",
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
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return "PROGRESS", "", current_solution, ""


class _PartialPipeline:
    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        return _Resp(content="<solution>draft</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>MINOR_FLAW</verdict>\n"
            "<verified_lemmas>Lemma: symmetry of equality.</verified_lemmas>\n"
            "<citation_review>NONE</citation_review>",
            VerificationDecision.MINOR_FLAW,
            "gap",
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
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        final_solution = "<done>done-part</done>\n<undone>todo-part</undone>"
        final_xml = (
            "<status>PROGRESS</status>\n"
            "<verdict>progress made</verdict>\n"
            f"<solution>{final_solution}</solution>"
        )
        return "PROGRESS", "progress made", final_solution, final_xml


class _PartialUndoneNonePipeline:
    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        return _Resp(content="<solution>draft</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>MINOR_FLAW</verdict>\n"
            "<verified_lemmas>Lemma: transitivity of equality.</verified_lemmas>\n"
            "<citation_review>NONE</citation_review>",
            VerificationDecision.MINOR_FLAW,
            "gap",
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
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        final_solution = "<done>done-part</done>\n<undone>无</undone>"
        final_xml = (
            "<status>PROGRESS</status>\n"
            "<verdict>progress made</verdict>\n"
            f"<solution>{final_solution}</solution>"
        )
        return "PROGRESS", "progress made", final_solution, final_xml


class _FailPipeline:
    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        raise TimeoutError("generator timeout")

    def call_verifier(self, problem_text: str, proof_text: str):
        raise AssertionError("verifier should not run on initial failure")

    def call_reviser(
        self,
        problem_text: str,
        previous_solution: str,
        verification_report: str,
        lemma_context_items: list[str] | None = None,
    ):
        raise AssertionError("reviser should not run on initial failure")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        raise AssertionError("final pipeline should not run on initial failure")


class _ParseRecoverPipeline:
    def __init__(self):
        self.generator_calls = 0
        self.verifier_calls = 0
        self.reviser_calls = 0

    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        self.generator_calls += 1
        return _Resp(content=f"<solution>candidate-{self.generator_calls}</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        self.verifier_calls += 1
        if self.verifier_calls == 1:
            return (
                "<verdict>MINOR_FLAW</verdict>\n"
                "<verification>Verifier output parsing failed previously; repair XML format and keep valid math.</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>NONE</citation_review>",
                VerificationDecision.MINOR_FLAW,
                "Verifier output parsing failed previously; repair XML format and keep valid math.",
                [],
                "phase1",
            )
        return (
            "<verdict>CORRECT</verdict>",
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
        self.reviser_calls += 1
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return "PROGRESS", "", current_solution, ""


class _ParseFailPipeline:
    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        return _Resp(content="<solution>candidate</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>MINOR_FLAW</verdict>\n"
            "<verification>Output is missing <solution> tag and must be fixed by Reviser.</verification>\n"
            "<verified_lemmas>NONE</verified_lemmas>\n"
            "<citation_review>NONE</citation_review>",
            VerificationDecision.MINOR_FLAW,
            "Output is missing <solution> tag and must be fixed by Reviser.",
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
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return "PROGRESS", "", current_solution, ""


def _read_state(runs_root: Path, problem_id: str) -> dict:
    state_path = runs_root / problem_id / "state.json"
    assert state_path.exists()
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_history_lines(runs_root: Path, problem_id: str) -> list[str]:
    history_path = runs_root / problem_id / "history.jsonl"
    assert history_path.exists()
    return [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_history_events(runs_root: Path, problem_id: str) -> list[dict]:
    return [json.loads(line) for line in _read_history_lines(runs_root, problem_id)]


def test_persist_success_state_and_history(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=2,
        pipeline=_SuccessPipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-success", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "SUCCESS"
    persisted = _read_state(runs_root, "p-success")
    assert persisted["status"] == "SUCCESS"
    history = _read_history_lines(runs_root, "p-success")
    assert len(history) >= 4


def test_persist_progress_state_and_history(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=1,
        pipeline=_PartialPipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-partial", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "PROGRESS"
    persisted = _read_state(runs_root, "p-partial")
    assert persisted["status"] == "PROGRESS"
    history = _read_history_lines(runs_root, "p-partial")
    assert len(history) >= 4


def test_progress_sets_final_answer_when_new_verified_lemma_exists(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=1,
        pipeline=_PartialUndoneNonePipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-partial-undone", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "PROGRESS"
    assert out.failure_reason == "max_turns_exhausted"
    assert out.final_answer is not None


def test_persist_failure_state_and_history(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=2,
        pipeline=_FailPipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-fail", problem_text="demo")

    with pytest.raises(RuntimeError, match="GENERATOR@turn0"):
        orchestrator.run(state)

    persisted = _read_state(runs_root, "p-fail")
    assert persisted["status"] == "RUNNING"
    history = _read_history_lines(runs_root, "p-fail")
    assert len(history) >= 1


def test_verifier_minor_flaw_recover_route(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    pipeline = _ParseRecoverPipeline()
    orchestrator = Orchestrator(
        max_turns=2,
        pipeline=pipeline,
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-parse-recover", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "SUCCESS"
    assert pipeline.generator_calls >= 1
    assert pipeline.reviser_calls >= 1
    events = _read_history_events(runs_root, "p-parse-recover")
    parse_events = [e for e in events if e.get("node") == "PARSE_ERROR"]
    assert not parse_events
    verifier_events = [e for e in events if e.get("node") == "VERIFIER"]
    assert verifier_events
    assert verifier_events[0].get("decision") == VerificationDecision.MINOR_FLAW.value


def test_verifier_minor_flaw_last_turn_goes_to_finalizer(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=1,
        pipeline=_ParseFailPipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-parse-fail", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "FAILED"
    assert out.failure_reason == "max_turns_exhausted"
    events = _read_history_events(runs_root, "p-parse-fail")
    parse_events = [e for e in events if e.get("node") == "PARSE_ERROR"]
    assert not parse_events


def test_citation_warning_soft_gate_route(tmp_path: Path) -> None:
    class _CitationPipeline:
        def call_generator(
            self,
            problem_text: str,
            lesson: str | None = None,
            lemma_context_items: list[str] | None = None,
        ):
            return _Resp(content="<solution>ok</solution>", reasoning_content="gen")

        def call_verifier(self, problem_text: str, proof_text: str):
            return (
                "<verdict>CORRECT</verdict>\n"
                "<verification>ok</verification>\n"
                "<verified_lemmas>NONE</verified_lemmas>\n"
                "<citation_review>{\"summary\":\"warn\",\"items\":[],\"fail_count\":2}</citation_review>",
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
            return _Resp(content=previous_solution, reasoning_content="rev")

        def call_final(self, problem_text: str, current_solution: str, last_verifier_decision: str, last_verification_report: str):
            return "PROGRESS", "", current_solution, ""

    runs_root = tmp_path / "runs"
    orchestrator = Orchestrator(
        max_turns=1,
        pipeline=_CitationPipeline(),
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-citation", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "SUCCESS"
    assert "Citation review reported 2 failed item" in (out.final_output or "")

    events = _read_history_events(runs_root, "p-citation")
    warning_events = [e for e in events if e.get("node") == "WARNING"]
    assert warning_events
    assert warning_events[0].get("warning_type") == "citation_review"


def test_route_critical_flaw_back_to_generator(tmp_path: Path) -> None:
    class _CriticalPipeline:
        def __init__(self):
            self.generator_calls = 0
            self.verifier_calls = 0

        def call_generator(
            self,
            problem_text: str,
            lesson: str | None = None,
            lemma_context_items: list[str] | None = None,
        ):
            self.generator_calls += 1
            return _Resp(content=f"<solution>candidate-{self.generator_calls}</solution>", reasoning_content="gen")

        def call_verifier(self, problem_text: str, proof_text: str):
            self.verifier_calls += 1
            if self.verifier_calls == 1:
                return (
                    "<verdict>CRITICAL_FLAW</verdict>\n"
                    "<verification>fatal issue</verification>\n"
                    "<verified_lemmas>NONE</verified_lemmas>\n"
                    "<citation_review>NONE</citation_review>",
                    VerificationDecision.CRITICAL_FLAW,
                    "fatal issue",
                    [],
                    "phase1",
                )
            return (
                "<verdict>CORRECT</verdict>\n"
                "<verification>fixed</verification>\n"
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
            return _Resp(content=previous_solution, reasoning_content="rev")

        def call_final(self, problem_text: str, current_solution: str, last_verifier_decision: str, last_verification_report: str):
            return "PROGRESS", "", current_solution, ""

    runs_root = tmp_path / "runs"
    pipeline = _CriticalPipeline()
    orchestrator = Orchestrator(
        max_turns=3,
        pipeline=pipeline,
        logger=_NoopLogger(),
        finalizer=_SimpleFinalizer(),
        runs_root=runs_root,
    )
    state = ProofState(problem_id="p-critical-route", problem_text="demo")

    out = orchestrator.run(state)

    assert out.status == "SUCCESS"
    assert pipeline.generator_calls >= 2

