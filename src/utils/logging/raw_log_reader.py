"""Raw JSONL 读取工具：供 WorklogBuilder 离线解析事件流。"""

import json
from pathlib import Path

RUNS_DIR = Path("runs")


def _normalize_runs_root(runs_root: Path | str = RUNS_DIR) -> Path:
	return Path(runs_root)


def resolve_run_dir(problem_id: str, runs_root: Path | str = RUNS_DIR) -> Path:
	"""返回 runs/{problem_id} 的标准路径。"""
	if not isinstance(problem_id, str) or not problem_id.strip():
		raise ValueError("problem_id must be a non-empty string")
	return _normalize_runs_root(runs_root) / problem_id.strip()


def resolve_run_log_path(problem_id: str, runs_root: Path | str = RUNS_DIR) -> Path:
	"""返回 runs/{problem_id}/history.jsonl 的标准路径。"""
	return resolve_run_dir(problem_id=problem_id, runs_root=runs_root) / "history.jsonl"


def resolve_run_artifact_path(
	problem_id: str,
	artifact_name: str,
	runs_root: Path | str = RUNS_DIR,
) -> Path:
	"""返回 runs/{problem_id}/artifact/{artifact_name} 的标准路径。"""
	if not isinstance(artifact_name, str) or not artifact_name.strip():
		raise ValueError("artifact_name must be a non-empty string")
	return resolve_run_dir(problem_id=problem_id, runs_root=runs_root) / "artifact" / artifact_name


def load_raw_events(problem_id: str, runs_root: Path | str = RUNS_DIR) -> list[dict]:
	"""读取并返回指定问题的 raw 事件列表。

	规则：
	1. 按文件行顺序返回事件，保持原始时序。
	2. 自动跳过空行。
	3. 若某行 JSON 损坏，抛出带行号的 ValueError，便于定位修复。
	"""
	filepath = resolve_run_log_path(problem_id=problem_id, runs_root=runs_root)
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
