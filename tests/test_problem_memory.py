from pathlib import Path

from src.memory.problem_memory import ProblemMemory
from src.memory.state import ProblemSnapshot, StageSnapshot, StateValidationError


def test_snapshot_problem_roundtrip() -> None:
    raw = {
        "problem_id": "demo-problem",
        "iteration_count": 2,
        "status": "RUNNING",
        "last_decision": "MINOR_FLAW",
        "stages": [
            {"stage_name": "GENERATOR", "status": "DONE", "last_error": None},
            {"stage_name": "VERIFIER", "status": "RUNNING", "last_error": None},
        ],
    }

    snapshot = ProblemSnapshot.from_dict(raw)
    assert snapshot.problem_id == "demo-problem"
    assert snapshot.iteration_count == 2
    assert snapshot.status == "RUNNING"
    assert snapshot.last_decision == "MINOR_FLAW"
    assert snapshot.to_dict() == raw


def test_snapshot_stage_roundtrip() -> None:
    raw = {"stage_name": "FINAL", "status": "PENDING", "last_error": None}
    stage = StageSnapshot.from_dict(raw)
    assert stage.stage_name == "FINAL"
    assert stage.status == "PENDING"
    assert stage.last_error is None
    assert stage.to_dict() == raw


def test_snapshot_problem_invalid_iteration_type() -> None:
    raw = {
        "problem_id": "demo-problem",
        "iteration_count": "2",
        "status": "RUNNING",
    }

    try:
        ProblemSnapshot.from_dict(raw)
        assert False, "Expected StateValidationError"
    except StateValidationError as exc:
        error = exc.to_dict()

    assert error["code"] == "invalid_field_type"
    assert error["field"] == "iteration_count"


def test_snapshot_problem_unknown_field() -> None:
    raw = {
        "problem_id": "demo-problem",
        "iteration_count": 1,
        "status": "RUNNING",
        "extra": "unexpected",
    }

    try:
        ProblemSnapshot.from_dict(raw)
        assert False, "Expected StateValidationError"
    except StateValidationError as exc:
        error = exc.to_dict()

    assert error["code"] == "unknown_field"
    assert error["field"] == "ProblemSnapshot"


def test_snapshot_problem_negative_iteration_value() -> None:
    raw = {
        "problem_id": "demo-problem",
        "iteration_count": -1,
        "status": "RUNNING",
    }

    try:
        ProblemSnapshot.from_dict(raw)
        assert False, "Expected StateValidationError"
    except StateValidationError as exc:
        error = exc.to_dict()

    assert error["code"] == "invalid_field_value"
    assert error["field"] == "iteration_count"


def test_problem_memory_init_dirs(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-001", runs_root=tmp_path / "runs")
    memory.init_dirs()

    assert memory.run_dir.exists()
    assert memory.artifact_dir.exists()
    assert memory.lemmas_dir.exists()
    assert memory.papers_dir.exists()
    assert memory.errors_dir.exists()


def test_problem_memory_state_save_load_merge(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-002", runs_root=tmp_path / "runs")
    state = {
        "problem_id": "p-002",
        "iteration_count": 0,
        "status": "RUNNING",
        "last_decision": None,
        "stages": [],
    }

    memory.save_state(state)
    loaded = memory.load_state()
    assert loaded is not None
    assert loaded.problem_id == "p-002"
    assert loaded.iteration_count == 0

    merged = memory.merge_state({"iteration_count": 1, "last_decision": "MINOR_FLAW"})
    assert merged.iteration_count == 1
    assert merged.last_decision == "MINOR_FLAW"


def test_problem_memory_event_roundtrip(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-003", runs_root=tmp_path / "runs")
    event = {
        "agent_node": "GENERATOR",
        "turn_id": 1,
        "timestamp": "2026-04-01T00:00:00Z",
        "content": "demo",
    }
    memory.append_event(event)
    events = memory.read_events()

    assert len(events) == 1
    assert events[0]["agent_node"] == "GENERATOR"
    assert events[0]["content"] == "demo"


def test_problem_memory_artifact_and_bibtex(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-004", runs_root=tmp_path / "runs")

    lemma_1 = memory.add_lemma("lemma one")
    lemma_2 = memory.add_lemma("lemma two")
    assert lemma_1.name == "001.md"
    assert lemma_2.name == "002.md"

    paper_1 = memory.add_paper("paper", filename="arXiv_2501.12345.md")
    paper_2 = memory.add_paper("paper", filename="arXiv_2501.12345.md")
    assert paper_1 == paper_2
    assert paper_1.read_text(encoding="utf-8").strip() == "paper"

    error_file = memory.add_error("bad path")
    assert error_file.name == "001.md"
    assert error_file.read_text(encoding="utf-8").strip() == "bad path"

    bib_path = memory.save_bibtex("@article{demo, title={Demo}}")
    assert bib_path.name == "citations.bib"
    assert bib_path.read_text(encoding="utf-8").strip() == "@article{demo, title={Demo}}"
