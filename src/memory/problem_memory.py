"""Per-problem memory manager for durable state, history, and artifacts."""

from __future__ import annotations

from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.memory.state import ProblemSnapshot


_CURRENT_PROBLEM_MEMORY: ContextVar["ProblemMemory | None"] = ContextVar(
    "current_problem_memory",
    default=None,
)


def set_current_problem_memory(memory: "ProblemMemory | None") -> None:
    _CURRENT_PROBLEM_MEMORY.set(memory)


def get_current_problem_memory() -> "ProblemMemory | None":
    return _CURRENT_PROBLEM_MEMORY.get()


class ProblemMemory:
    """Manage persistence under runs/{problem_id}.

    Layout:
      runs/{problem_id}/
        state.json
        history.jsonl
        artifact/
          lemmas/
          papers/
          errors/
          citations.bib
    """

    def __init__(self, problem_id: str, runs_root: Path | str = "runs"):
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")

        self.problem_id = problem_id.strip()
        self.runs_root = Path(runs_root)
        self.run_dir = self.runs_root / self.problem_id
        self.state_path = self.run_dir / "state.json"
        self.history_path = self.run_dir / "history.jsonl"

        self.artifact_dir = self.run_dir / "artifact"
        self.lemmas_dir = self.artifact_dir / "lemmas"
        self.papers_dir = self.artifact_dir / "papers"
        self.errors_dir = self.artifact_dir / "errors"
        self.citations_bib_path = self.artifact_dir / "citations.bib"

    def init_dirs(self) -> None:
        self.lemmas_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)

    def save_state(self, state: ProblemSnapshot | dict[str, Any]) -> ProblemSnapshot:
        self.init_dirs()
        snapshot = state if isinstance(state, ProblemSnapshot) else ProblemSnapshot.from_dict(state)
        serialized = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        self._atomic_write_text(self.state_path, serialized + "\n")
        return snapshot

    def load_state(self) -> ProblemSnapshot | None:
        if not self.state_path.exists():
            return None
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return ProblemSnapshot.from_dict(data)

    def merge_state(self, patch: dict[str, Any]) -> ProblemSnapshot:
        if not isinstance(patch, dict):
            raise TypeError("patch must be a dict")
        current = self.load_state()
        merged = current.to_dict() if current else {}
        merged.update(patch)
        snapshot = ProblemSnapshot.from_dict(merged)
        return self.save_state(snapshot)

    def append_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        required = ("agent_node", "turn_id", "timestamp")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"event missing required keys: {missing}")

        self.init_dirs()
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with open(self.history_path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.history_path} at line {line_no}: {exc.msg}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Invalid event object in {self.history_path} at line {line_no}: not a JSON object"
                    )
                events.append(obj)
        return events

    @staticmethod
    def _next_numeric_name(folder: Path) -> str:
        max_idx = 0
        for item in folder.glob("*.md"):
            stem = item.stem
            if stem.isdigit():
                max_idx = max(max_idx, int(stem))
        return f"{max_idx + 1:03d}.md"

    def _save_markdown(self, folder: Path, content: str, filename: str | None = None) -> Path:
        self.init_dirs()
        target_name = filename or self._next_numeric_name(folder)
        target = folder / target_name
        normalized = (content or "").rstrip() + "\n"

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if existing == normalized:
                return target

        self._atomic_write_text(target, normalized)
        return target

    def add_lemma(self, content: str, filename: str | None = None) -> Path:
        return self._save_markdown(self.lemmas_dir, content, filename=filename)

    def add_paper(self, content: str, filename: str | None = None) -> Path:
        return self._save_markdown(self.papers_dir, content, filename=filename)

    def add_error(self, content: str, filename: str | None = None) -> Path:
        return self._save_markdown(self.errors_dir, content, filename=filename)

    def save_bibtex(self, bibtex: str) -> Path:
        self.init_dirs()
        normalized = (bibtex or "").rstrip() + "\n"
        if self.citations_bib_path.exists():
            existing = self.citations_bib_path.read_text(encoding="utf-8")
            if existing == normalized:
                return self.citations_bib_path
        self._atomic_write_text(self.citations_bib_path, normalized)
        return self.citations_bib_path
