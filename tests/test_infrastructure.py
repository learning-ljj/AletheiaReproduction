from pathlib import Path

from src.utils.logger import append_raw_event
from src.utils.raw_log_reader import load_raw_events, resolve_run_log_path
from src.utils.worklog_builder import WorklogBuilder


def test_log_path_logger_writes_to_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    payload = {
        "agent_node": "GENERATOR",
        "turn_id": 0,
        "timestamp": "2026-04-01T00:00:00Z",
        "content": "demo",
    }

    append_raw_event(problem_id="p-log", payload=payload, runs_root=runs_root)

    log_path = resolve_run_log_path(problem_id="p-log", runs_root=runs_root)
    assert log_path.exists()
    assert "data" not in str(log_path)



def test_log_path_reader_uses_runs_layout(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    append_raw_event(
        problem_id="p-read",
        payload={
            "agent_node": "VERIFIER",
            "turn_id": 1,
            "timestamp": "2026-04-01T00:00:01Z",
            "decision": "MINOR_FLAW",
        },
        runs_root=runs_root,
    )

    events = load_raw_events(problem_id="p-read", runs_root=runs_root)
    assert len(events) == 1
    assert events[0]["agent_node"] == "VERIFIER"



def test_log_path_worklog_builder_consumes_runs_stream(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    problem_id = "p-worklog"

    append_raw_event(
        problem_id=problem_id,
        payload={
            "agent_node": "RUN_START",
            "turn_id": -1,
            "timestamp": "2026-04-01T00:00:00Z",
            "problem_text": "demo",
            "ground_truth": "42",
        },
        runs_root=runs_root,
    )
    append_raw_event(
        problem_id=problem_id,
        payload={
            "agent_node": "FINAL",
            "turn_id": 1,
            "timestamp": "2026-04-01T00:00:02Z",
            "final_output": "final demo",
        },
        runs_root=runs_root,
    )

    run_jsonl_path = resolve_run_log_path(problem_id=problem_id, runs_root=runs_root)
    output_md = tmp_path / "worklog.md"

    builder = WorklogBuilder(llm_client=None, llm_config=None)
    builder.build_problem_worklog(str(run_jsonl_path), str(output_md))

    assert output_md.exists()
    text = output_md.read_text(encoding="utf-8")
    assert problem_id in text
    assert "final demo" in text
