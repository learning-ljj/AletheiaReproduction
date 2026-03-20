"""Raw JSONL 读取工具：供 WorklogBuilder 离线解析事件流。"""

import json
from pathlib import Path


def load_raw_events(problem_id: str, log_dir: Path = Path("data/logs")) -> list[dict]:
    """读取并返回指定问题的 raw 事件列表。

    规则：
    1. 按文件行顺序返回事件，保持原始时序。
    2. 自动跳过空行。
    3. 若某行 JSON 损坏，抛出带行号的 ValueError，便于定位修复。
    """
    filepath = log_dir / f"{problem_id}.jsonl"
    if not filepath.exists():
        return []

    events: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {filepath}:{line_no}: {exc.msg}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Invalid event object at {filepath}:{line_no}: not a JSON object")

            events.append(obj)

    return events
