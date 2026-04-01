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
