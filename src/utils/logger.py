"""JSONL 日志持久化。"""

import json
from pathlib import Path

# 默认日志目录
LOG_DIR = Path("data/logs")


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


