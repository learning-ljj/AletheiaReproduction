"""Citation existence checker tool.

只检查引用路径是否能解析到文件，并返回该文件的全文。
"""

from __future__ import annotations

from pathlib import Path

from src.memory.problem_memory import ProblemMemory


def _resolve_citation_path(problem_memory: ProblemMemory, cite_path: str) -> Path:
    raw = Path(cite_path)
    if raw.is_absolute():
        return raw

    normalized = cite_path.replace("\\", "/")
    if normalized.startswith("runs/"):
        return Path(normalized)

    return (problem_memory.run_dir / raw).resolve()


def _read_full_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def review_citation(
    *,
    problem_memory: ProblemMemory,
    cites: list[str],
) -> dict:
    """检查引用文件是否存在，并返回对应文件全文。"""
    normalized_cites = [(item or "").strip() for item in cites if (item or "").strip()]
    resolved_items: list[dict[str, object]] = []
    missing_cites: list[str] = []

    for cite_path in normalized_cites:
        resolved_path = _resolve_citation_path(problem_memory, cite_path)
        exists = resolved_path.exists() and resolved_path.is_file()
        content = _read_full_text(resolved_path) if exists else None

        if not exists:
            missing_cites.append(cite_path)

        resolved_items.append(
            {
                "cite": cite_path,
                "resolved_path": str(resolved_path),
                "exists": exists,
                "content": content,
            }
        )

    return {
        "all_exist": not missing_cites,
        "summary": (
            f"Reviewed {len(normalized_cites)} citation path(s): "
            f"{len(normalized_cites) - len(missing_cites)} exist, {len(missing_cites)} missing."
        ),
        "cites": normalized_cites,
        "items": resolved_items,
        "missing_cites": missing_cites,
        "existing_count": len(normalized_cites) - len(missing_cites),
        "missing_count": len(missing_cites),
    }
