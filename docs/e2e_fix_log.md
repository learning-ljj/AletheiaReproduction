# E2E Fix Log

## Context
- Phase: F64 final gate and defect closure
- Date: 2026-04-01
- Scope: full test gate, proofbench smoke benchmark, and path consistency fixes

## Gate Results

1. Full test suite
- Command: `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest -q`
- Result: PASS
- Evidence: `46 passed in 0.82s` (initial gate), `46 passed in 0.70s` (after final fixes)

2. ProofBench smoke benchmark
- Command: `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe scripts/run_imobench.py --dataset proofbench --count 2 --max-turns 2`
- Result: PASS
- Output file: `runs/benchmarks/imobench_proofbench_20260401_131327.json`
- Summary snapshot:
  - `total: 2`
  - `format_complete_rate: 1.0`
  - `correct_verdict_rate: 1.0`
  - `has_final_answer_rate: 1.0`
  - `error_count: 0`
  - `lemma_accept_rate: 0.0`
  - `citation_warning_rate: 0.0`

## Defects Closed

1. Benchmark/log path migration completed (runs-only)
- Files:
  - `scripts/run_imobench.py`
  - `src/utils/raw_log_reader.py`
- Fix:
  - benchmark now reads per-run raw events from `runs/{run_id}/history.jsonl`
  - added metrics: `lemma_accept_rate`, `citation_warning_rate`
  - worklog generation switched to `runs/{run_id}/artifact/worklog.md`
  - default benchmark output switched to `runs/benchmarks`

2. Realtime E2E monitor migrated to runs stream
- File: `tests/realtime_e2e_monitor.py`
- Fix:
  - monitoring artifacts moved to `runs/monitoring`
  - event polling switched to `runs/{run_id}/history.jsonl`
  - monitor now generates worklog from runs stream (`worklog.monitor.md`)
- Acceptance evidence:
  - command: `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe tests/realtime_e2e_monitor.py --input tasks_v1.md --max-turns 2`
  - tracking: `runs/monitoring/realtime_tasks_v1_20260401_130359.tracking.md`
  - final summary includes `final_status`, `warning_count`, `bug_count`

3. CLI path consistency defect fixed during F64
- File: `main.py`
- Problem:
  - CLI still referenced `data/logs` for worklog source/output and JSONL print message.
- Fix:
  - `_maybe_build_worklog` now resolves logs from `runs` using `resolve_run_log_path`
  - default worklog destination switched to `runs/{run_id}/artifact/worklog.md`
  - final JSONL path print switched to runs path

## Notes
- Benchmark runtime remains model-latency sensitive; pass/fail should be judged by output artifacts and summary JSON, not wall-clock duration.
- Current gate state is green and reproducible with the commands above.
