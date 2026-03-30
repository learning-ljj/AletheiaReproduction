"""JSONL 日志持久化。"""

import json
from pathlib import Path

# 默认日志目录
LOG_DIR = Path("data/logs")
ARTIFACT_DIR = Path("artifact")


def append_raw_event(problem_id: str, payload: dict, log_dir: Path = LOG_DIR) -> None:
    """写入一条 raw 事件日志（JSONL）。

    必须包含字段：agent_node, turn_id, timestamp。
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    required_keys = ("agent_node", "turn_id", "timestamp")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"raw payload missing required keys: {missing_keys}")

    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{problem_id}.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_final_output_markdown(problem_id: str, final_output: str, artifact_dir: Path = ARTIFACT_DIR) -> Path:
    """将 final_output 原文保存为 Markdown，文件名与 run_id（problem_id）一致。"""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifact_dir / f"{problem_id}.md"
    target_path.write_text((final_output or "").strip() + "\n", encoding="utf-8")
    return target_path


