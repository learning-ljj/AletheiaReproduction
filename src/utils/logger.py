"""JSONL 日志持久化。"""

import json
from pathlib import Path

# 默认运行目录
RUNS_DIR = Path("runs")


def _resolve_run_dir(problem_id: str, runs_root: Path) -> Path:
    return runs_root / problem_id


def append_raw_event(problem_id: str, payload: dict, runs_root: Path = RUNS_DIR) -> None:
    """写入一条 raw 事件日志（JSONL）。

    必须包含字段：agent_node, turn_id, timestamp。
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    required_keys = ("agent_node", "turn_id", "timestamp")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"raw payload missing required keys: {missing_keys}")

    run_dir = _resolve_run_dir(problem_id, runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    filepath = run_dir / "history.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_final_output_markdown(problem_id: str, final_output: str, runs_root: Path = RUNS_DIR) -> Path:
    """将 final_output 原文保存为 Markdown，文件名与 run_id（problem_id）一致。"""
    artifact_dir = _resolve_run_dir(problem_id, runs_root) / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifact_dir / "final_output.md"
    target_path.write_text((final_output or "").strip() + "\n", encoding="utf-8")
    return target_path


