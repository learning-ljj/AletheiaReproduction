"""Search utilities for SearcherAgent retrieval chain."""

from __future__ import annotations

import re
from typing import Callable


def normalize_title(title: str) -> str:
    text = (title or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sanitize_token(text: str, *, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", (text or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def paper_identity(paper: dict) -> tuple[str, str]:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return "doi", doi

    arxiv_id = (paper.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return "arxiv", arxiv_id

    title_key = normalize_title(str(paper.get("title") or ""))
    return "title", title_key


def dedup_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate papers with DOI > arXiv ID > normalized title priority."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for paper in papers:
        key = paper_identity(paper)
        if not key[1]:
            # Empty identity is treated as low-quality record and skipped.
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(paper)
    return out


def expand_queries(query: str | None = None, query_bundle: list[str] | None = None) -> list[str]:
    candidates = []
    if query:
        candidates.append(query)
    candidates.extend(query_bundle or [])

    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = (item or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def multi_source_search(
    queries: list[str],
    source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None,
    limit_per_query: int = 10,
) -> list[dict]:
    """Aggregate search results from multiple pluggable sources."""
    if not source_handlers:
        return []

    results: list[dict] = []
    for source_name, handler in source_handlers.items():
        for query in queries:
            try:
                items = handler(query, limit_per_query) or []
            except Exception:
                items = []
            for item in items:
                record = dict(item)
                record.setdefault("source", source_name)
                results.append(record)
    return results


def build_paper_filename(paper: dict) -> str:
    doi = (paper.get("doi") or "").strip()
    if doi:
        return f"doi_{_sanitize_token(doi, fallback='paper')}.md"

    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"arXiv_{_sanitize_token(arxiv_id, fallback='paper')}.md"

    title = normalize_title(str(paper.get("title") or ""))
    return f"title_{_sanitize_token(title[:80], fallback='paper')}.md"


def build_paper_markdown(paper: dict) -> str:
    title = str(paper.get("title") or "Unknown Title")
    abstract = str(paper.get("abstract") or "")
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        author_line = "; ".join(str(a) for a in authors)
    else:
        author_line = str(authors)

    summary = abstract.strip()[:240] if abstract else f"Paper summary for: {title}"
    layer2 = abstract.strip() or "No extracted body available."

    lines = [
        "---",
        f"title: {title}",
        f"summary: {summary}",
        f"doi: {paper.get('doi') or ''}",
        f"arxiv_id: {paper.get('arxiv_id') or ''}",
        "---",
        "",
        "## Layer2-Extracted",
        layer2,
        "",
        "## Layer3-Source",
        f"source: {paper.get('source') or ''}",
        f"url: {paper.get('url') or ''}",
        f"authors: {author_line}",
    ]
    return "\n".join(lines)
