"""Run one E2E problem with real-time tracking and UTF-8-safe logs.

Usage:
    .venv\\Scripts\\python.exe scripts/realtime_e2e_monitor.py \
      --input data/e2e_inputs/imo-bench-algebra-001.txt \
      --max-turns 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _find_run_id(stdout_text: str) -> str | None:
    m = re.search(r"^>>> Run ID:\s+(.+)$", stdout_text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _collect_signals(stdout_text: str, stderr_text: str, events: list[dict]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    bugs: list[str] = []

    blob = "\n".join([stdout_text, stderr_text])
    for kw in ["timeout", "timed out", "retrying", "missing required", "parse_error"]:
        if kw.lower() in blob.lower():
            warnings.append(f"text-signal: {kw}")

    for ev in events:
        if ev.get("agent_node") == "FINAL" and ev.get("failure_reason"):
            fr = str(ev.get("failure_reason"))
            if fr in {"timeout", "parse_error", "llm_failure", "tool_failure"}:
                warnings.append(f"final-failure-reason: {fr}")

        if ev.get("agent_node") == "VERIFIER":
            traces = ev.get("tool_calls_trace") or []
            for i, t in enumerate(traces, start=1):
                res = str(t.get("result") or "")
                if "Traceback" in res or "AttributeError" in res or "exit_code: 1" in res:
                    bugs.append(
                        f"verifier_turn={ev.get('turn_id')} tool#{i} name={t.get('name')} failed"
                    )

    warnings = sorted(set(warnings))
    bugs = sorted(set(bugs))
    return warnings, bugs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-time E2E monitor runner")
    p.add_argument("--input", required=True, help="Problem input txt path")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--tail-lines", type=int, default=40)
    p.add_argument("--out-prefix", default=None, help="Optional output prefix under data/logs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    logs_dir = repo / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out_prefix or f"realtime_{Path(args.input).stem}_{stamp}"

    stdout_path = logs_dir / f"{base}.console.txt"
    stderr_path = logs_dir / f"{base}.err.txt"
    tracking_path = logs_dir / f"{base}.tracking.md"
    worklog_path = logs_dir / f"{base}.md"

    cmd = [
        str(repo / ".venv" / "Scripts" / "python.exe"),
        "-u",
        str(repo / "main.py"),
        args.input,
        "--max-turns",
        str(args.max_turns),
        "--generate-worklog",
        "--worklog-path",
        str(worklog_path),
    ]

    with tracking_path.open("w", encoding="utf-8", newline="\n") as tr:
        tr.write(f"# Realtime E2E Tracking - {base}\n")
        tr.write(f"- started_at: {_now()}\n")
        tr.write(f"- input: {args.input}\n")
        tr.write(f"- command: {' '.join(cmd)}\n")
        tr.write(f"- stdout: {stdout_path}\n")
        tr.write(f"- stderr: {stderr_path}\n")
        tr.write(f"- tracking: {tracking_path}\n")
        tr.write(f"- worklog_target: {worklog_path}\n")

    with stdout_path.open("w", encoding="utf-8", newline="\n") as out, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as err:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=out,
            stderr=err,
            env=os.environ.copy(),
        )

    poll_id = 0
    run_id: str | None = None
    jsonl_path: Path | None = None

    while proc.poll() is None:
        time.sleep(max(2, args.poll_seconds))
        poll_id += 1

        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""

        if run_id is None:
            run_id = _find_run_id(stdout_text)
            if run_id:
                jsonl_path = logs_dir / f"{run_id}.jsonl"

        events = _load_jsonl(jsonl_path) if jsonl_path else []
        warnings, bugs = _collect_signals(stdout_text, stderr_text, events)

        with tracking_path.open("a", encoding="utf-8", newline="\n") as tr:
            tr.write("\n")
            tr.write(f"## Poll #{poll_id} @ {_now()}\n")
            tr.write(f"- alive: true\n")
            tr.write(f"- stdout_bytes: {stdout_path.stat().st_size if stdout_path.exists() else 0}\n")
            tr.write(f"- stderr_bytes: {stderr_path.stat().st_size if stderr_path.exists() else 0}\n")
            tr.write(f"- run_id: {run_id or '(pending)'}\n")
            tr.write(f"- jsonl: {jsonl_path.name if jsonl_path and jsonl_path.exists() else '(pending)'}\n")
            tr.write(f"- warning_count: {len(warnings)}\n")
            tr.write(f"- bug_count: {len(bugs)}\n")
            tr.write("```stdout\n")
            tr.write(_tail_text(stdout_path, args.tail_lines))
            tr.write("\n```\n")
            tr.write("```stderr\n")
            tr.write(_tail_text(stderr_path, args.tail_lines))
            tr.write("\n```\n")
            if jsonl_path and jsonl_path.exists():
                tr.write("```jsonl\n")
                tr.write(_tail_text(jsonl_path, min(args.tail_lines, 20)))
                tr.write("\n```\n")

    exit_code = proc.returncode
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    if run_id is None:
        run_id = _find_run_id(stdout_text)
        if run_id:
            jsonl_path = logs_dir / f"{run_id}.jsonl"

    events = _load_jsonl(jsonl_path) if jsonl_path else []
    warnings, bugs = _collect_signals(stdout_text, stderr_text, events)

    with tracking_path.open("a", encoding="utf-8", newline="\n") as tr:
        tr.write("\n")
        tr.write("## Final Summary\n")
        tr.write(f"- finished_at: {_now()}\n")
        tr.write(f"- exit_code: {exit_code}\n")
        tr.write(f"- run_id: {run_id or '(unknown)'}\n")
        tr.write(f"- jsonl: {jsonl_path.name if jsonl_path and jsonl_path.exists() else '(missing)'}\n")

        final_event = next((e for e in reversed(events) if e.get("agent_node") == "FINAL"), None)
        if final_event:
            tr.write(f"- final_status: {final_event.get('status')}\n")
            tr.write(f"- failure_reason: {final_event.get('failure_reason')}\n")

        tr.write(f"- warning_count: {len(warnings)}\n")
        for w in warnings:
            tr.write(f"  - {w}\n")
        tr.write(f"- bug_count: {len(bugs)}\n")
        for b in bugs:
            tr.write(f"  - {b}\n")

    print(f"TRACKING={tracking_path}")
    print(f"STDOUT={stdout_path}")
    print(f"STDERR={stderr_path}")
    print(f"WORKLOG={worklog_path}")
    if jsonl_path and jsonl_path.exists():
        print(f"JSONL={jsonl_path}")
    print(f"EXIT_CODE={exit_code}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
