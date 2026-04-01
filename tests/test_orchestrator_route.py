import json
from pathlib import Path

from src.core.orchestrator import Orchestrator
from src.core.state import ProofState, RunStatus, VerificationDecision


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _NoopLogger:
    def append_raw_event(self, problem_id: str, payload: dict) -> None:
        # A13 之后，raw event 由 ProblemMemory 负责，这里保持兼容占位。
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
    ) -> str:
        status = "success" if success else ("partial" if partial else "failed")
        return f"status={status};reason={failure_reason};solution={solution_text};assessment={assessment_output}"


class _SuccessPipeline:
    def call_generator(self, problem_text: str, lesson: str | None = None):
        return _Resp(content="<solution>ok</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>CORRECT</verdict>",
            VerificationDecision.CORRECT,
            "",
            [],
            "phase1",
        )

    def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
        return _Resp(content=previous_solution, reasoning_content="rev")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return RunStatus.PARTIAL.value, "", current_solution, ""


class _PartialPipeline:
    def call_generator(self, problem_text: str, lesson: str | None = None):
        return _Resp(content="<solution>draft</solution>", reasoning_content="gen")

    def call_verifier(self, problem_text: str, proof_text: str):
        return (
            "<verdict>MINOR_FLAW</verdict>",
            VerificationDecision.MINOR_FLAW,
            "gap",
            [],
            "phase1",
        )

    def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
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
            "<status>PARTIAL_PROGRESS</status>\n"
            "<verdict>progress made</verdict>\n"
            f"<solution>{final_solution}</solution>"
        )
        return RunStatus.PARTIAL.value, "progress made", final_solution, final_xml


class _FailPipeline:
    def call_generator(self, problem_text: str, lesson: str | None = None):
        raise TimeoutError("generator timeout")

    def call_verifier(self, problem_text: str, proof_text: str):
        raise AssertionError("verifier should not run on initial failure")

    def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
        raise AssertionError("reviser should not run on initial failure")

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        raise AssertionError("final pipeline should not run on initial failure")


def _read_state(runs_root: Path, problem_id: str) -> dict:
    state_path = runs_root / problem_id / "state.json"
    assert state_path.exists()
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_history_lines(runs_root: Path, problem_id: str) -> list[str]:
    history_path = runs_root / problem_id / "history.jsonl"
    assert history_path.exists()
    return [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]


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

    assert out.status == RunStatus.SUCCESS
    persisted = _read_state(runs_root, "p-success")
    assert persisted["status"] == RunStatus.SUCCESS.value
    history = _read_history_lines(runs_root, "p-success")
    assert len(history) >= 4


def test_persist_partial_state_and_history(tmp_path: Path) -> None:
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

    assert out.status == RunStatus.PARTIAL
    persisted = _read_state(runs_root, "p-partial")
    assert persisted["status"] == RunStatus.PARTIAL.value
    history = _read_history_lines(runs_root, "p-partial")
    assert len(history) >= 4


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

    out = orchestrator.run(state)

    assert out.status == RunStatus.FAILED
    persisted = _read_state(runs_root, "p-fail")
    assert persisted["status"] == RunStatus.FAILED.value
    history = _read_history_lines(runs_root, "p-fail")
    assert len(history) >= 2
