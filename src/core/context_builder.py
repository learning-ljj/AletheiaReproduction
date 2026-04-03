"""Context assembly helpers for stage prompts."""

from __future__ import annotations

from pathlib import Path

from src.memory.problem_memory import ProblemMemory


class ContextBuilder:
    """Build lightweight per-stage context from ProblemMemory artifacts."""

    def __init__(self, problem_memory: ProblemMemory):
        self.problem_memory = problem_memory

    def _collect_recent_error_hints(self, limit: int = 3) -> list[str]:
        hints: list[str] = []
        error_files = sorted(self.problem_memory.errors_dir.glob("*.md"))
        for path in error_files[-limit:]:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if not first_line:
                continue
            rel = path.relative_to(self.problem_memory.run_dir).as_posix()
            hints.append(f"{first_line} [path:{rel}]")
        return hints

    def build_generator_layer1_context(self, summary_limit: int = 12, error_limit: int = 3) -> list[str]:
        summaries = self.problem_memory.list_layer1_summaries(limit=summary_limit)
        error_hints = self._collect_recent_error_hints(limit=error_limit)
        if error_hints:
            summaries.extend([f"ErrorHint: {item}" for item in error_hints])
        return summaries
