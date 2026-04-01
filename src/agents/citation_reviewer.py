"""Citation reviewer sub-agent for cite path and claim consistency checks."""

from __future__ import annotations

from pathlib import Path

from src.memory.problem_memory import ProblemMemory


class CitationReviewerAgent:
    """Review citation paths and basic claim consistency."""

    def __init__(self, *, problem_memory: ProblemMemory | None = None):
        self.problem_memory = problem_memory

    def _resolve_cite_path(self, cite_path: str) -> Path:
        raw = Path(cite_path)
        if raw.is_absolute():
            return raw

        as_posix = cite_path.replace("\\", "/")
        if as_posix.startswith("runs/"):
            return Path(as_posix)

        if self.problem_memory is not None:
            return (self.problem_memory.run_dir / raw).resolve()

        return raw

    def _review_one(self, cite_path: str, claim_span: str | None = None) -> dict:
        resolved = self._resolve_cite_path(cite_path)
        exists = resolved.exists() and resolved.is_file()
        if not exists:
            return {
                "cite": cite_path,
                "resolved_path": str(resolved),
                "passed": False,
                "reason": "PATH_NOT_FOUND",
            }

        content = resolved.read_text(encoding="utf-8")
        if claim_span:
            if claim_span.strip() and claim_span not in content:
                return {
                    "cite": cite_path,
                    "resolved_path": str(resolved),
                    "passed": False,
                    "reason": "CLAIM_NOT_SUPPORTED",
                }

        return {
            "cite": cite_path,
            "resolved_path": str(resolved),
            "passed": True,
            "reason": "OK",
        }

    def review(self, *, cites: list[str], claim_spans: list[str] | None = None) -> dict:
        if not cites:
            return {
                "summary": "No citations to review.",
                "items": [],
                "fail_count": 0,
            }

        spans = claim_spans or []
        items: list[dict] = []
        fail_count = 0

        for idx, cite in enumerate(cites):
            claim_span = spans[idx] if idx < len(spans) else None
            item = self._review_one(cite, claim_span)
            if not item["passed"]:
                fail_count += 1
            items.append(item)

        passed = len(items) - fail_count
        summary = f"Reviewed {len(items)} citations: {passed} passed, {fail_count} failed."
        return {
            "summary": summary,
            "items": items,
            "fail_count": fail_count,
        }
