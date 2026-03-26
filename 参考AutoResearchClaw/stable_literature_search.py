"""Low-coupling literature search module inspired by ResearchClaw.

This module keeps only the stable retrieval core:
1. Normalize and expand queries.
2. Search multiple academic sources.
3. Cache per source for graceful degradation.
4. Deduplicate by DOI, arXiv id, then normalized title.

It only depends on the standard library plus `shared_models.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request
import argparse
import hashlib
import json
import re
import time

from docs.analysis.migration_core.shared_models import Author, Paper


# Keep cache local to the module so migration does not depend on repo-specific dirs.
CACHE_ROOT = Path(".migration_cache") / "literature"

# Search suffixes copied from the repo idea: they improve retrieval recall.
SEARCH_SUFFIXES = [
    "survey",
    "review",
    "benchmark",
    "comparison",
    "systematic review",
]

# A tiny stopword list is enough for query shortening.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}

# Cache TTL intentionally mirrors the original repo's strategy.
TTL_BY_SOURCE = {
    "openalex": timedelta(days=3),
    "semantic_scholar": timedelta(days=3),
    "arxiv": timedelta(days=1),
}

# A small per-source request spacing reduces rate-limit bursts.
REQUEST_INTERVAL_SECONDS = {
    "openalex": 1.0,
    "semantic_scholar": 1.5,
    "arxiv": 3.0,
}

# Simple in-memory request bookkeeping.
_LAST_REQUEST_AT: dict[str, float] = {}

# Simple circuit breaker state for stricter sources.
_BREAKER_STATE: dict[str, dict[str, float | int]] = {
    "semantic_scholar": {"failures": 0, "open_until": 0.0},
    "arxiv": {"failures": 0, "open_until": 0.0},
}


@dataclass(slots=True)
class SearchConfig:
    """Search configuration kept agent-framework neutral."""

    sources: list[str]
    limit_per_query: int = 10
    year_min: int | None = None
    semantic_scholar_api_key: str | None = None


def utcnow() -> datetime:
    """Return UTC now in one place for easier testing."""
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy equality checks."""
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_keywords(text: str, max_keywords: int = 6) -> list[str]:
    """Keep only informative words from a long topic sentence."""
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    keywords = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return keywords[:max_keywords]


def shorten_query(query: str, max_keywords: int = 6) -> str:
    """Convert a long topic description into a short API-friendly query."""
    query = query.strip()
    suffix = ""
    core = query
    for known_suffix in SEARCH_SUFFIXES:
        if query.lower().endswith(known_suffix):
            suffix = known_suffix
            core = query[: -len(known_suffix)].strip()
            break
    shortened = " ".join(extract_keywords(core, max_keywords=max_keywords))
    return f"{shortened} {suffix}".strip() or query


def expand_queries(queries: list[str], topic: str) -> list[str]:
    """Generate a compact set of useful search variants."""
    seeds = list(queries) if queries else [topic]
    expanded: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        candidates = [
            seed,
            shorten_query(seed),
            shorten_query(topic),
        ]
        for suffix in SEARCH_SUFFIXES[:4]:
            candidates.append(f"{shorten_query(seed)} {suffix}".strip())
        for candidate in candidates:
            normalized = normalize_text(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                expanded.append(candidate)
    return expanded


def ensure_cache_dir() -> None:
    """Create the local cache directory lazily."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def cache_key(query: str, source: str, limit: int, year_min: int | None) -> str:
    """Hash parameters so cache file names stay short and portable."""
    payload = json.dumps(
        {"query": query, "source": source, "limit": limit, "year_min": year_min},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(query: str, source: str, limit: int, year_min: int | None) -> Path:
    """Return the on-disk cache path for one query-source pair."""
    return CACHE_ROOT / f"{cache_key(query, source, limit, year_min)}.json"


def cache_get(query: str, source: str, limit: int, year_min: int | None) -> list[Paper] | None:
    """Read cache if it exists and is still within the source TTL."""
    # 先确保缓存目录存在，这样后续路径计算不会出错。
    ensure_cache_dir()
    # 同一个 query/source/limit/year_min 会映射到同一个缓存文件。
    path = cache_path(query, source, limit, year_min)
    if not path.exists():
        return None
    try:
        # 直接把磁盘上的 JSON 读出来；坏文件按缓存 miss 处理。
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    created_at = payload.get("created_at")
    if not isinstance(created_at, str):
        return None
    try:
        created_dt = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    ttl = TTL_BY_SOURCE.get(source, timedelta(days=1))
    # 超过 TTL 的缓存不再使用，避免旧结果污染检索。
    if utcnow() - created_dt > ttl:
        return None
    rows = payload.get("papers", [])
    if not isinstance(rows, list):
        return None
    return [Paper.from_dict(row) for row in rows if isinstance(row, dict)]


def cache_put(query: str, source: str, limit: int, year_min: int | None, papers: list[Paper]) -> None:
    """Write successful online results to cache for later fallback."""
    ensure_cache_dir()
    path = cache_path(query, source, limit, year_min)
    payload = {
        "created_at": utcnow().isoformat(),
        "papers": [paper.to_dict() for paper in papers],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_rate_limit(source: str) -> None:
    """Sleep between requests so we do not hit APIs too aggressively."""
    interval = REQUEST_INTERVAL_SECONDS.get(source, 0.0)
    if interval <= 0:
        return
    last_at = _LAST_REQUEST_AT.get(source, 0.0)
    elapsed = time.time() - last_at
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _LAST_REQUEST_AT[source] = time.time()


def breaker_is_open(source: str) -> bool:
    """Return True when a source is temporarily disabled after many 429s."""
    state = _BREAKER_STATE.get(source)
    if not state:
        return False
    return time.time() < float(state.get("open_until", 0.0))


def breaker_record_success(source: str) -> None:
    """Reset failure counters after a successful request."""
    state = _BREAKER_STATE.get(source)
    if state:
        state["failures"] = 0
        state["open_until"] = 0.0


def breaker_record_failure(source: str, retry_after_seconds: float = 60.0) -> None:
    """Open the circuit after repeated failures to avoid hammering the API."""
    state = _BREAKER_STATE.get(source)
    if not state:
        return
    state["failures"] = int(state.get("failures", 0)) + 1
    if int(state["failures"]) >= 3:
        state["open_until"] = time.time() + retry_after_seconds


def request_json(url: str, *, headers: dict[str, str] | None = None, source: str, retries: int = 2) -> Any:
    """Perform a JSON HTTP request with basic retry and breaker support."""
    # 如果熔断器处于打开状态，直接失败，避免继续打爆 API。
    if breaker_is_open(source):
        raise RuntimeError(f"{source} circuit breaker is open")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # 先做最小限流，再发请求。
            wait_for_rate_limit(source)
            req = request.Request(url, headers=headers or {})
            with request.urlopen(req, timeout=20) as response:
                data = response.read().decode("utf-8")
            # 一旦请求成功，就重置失败计数。
            breaker_record_success(source)
            return json.loads(data)
        except error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                # 429 表示触发限流，记录失败并指数式退避。
                breaker_record_failure(source)
                time.sleep(min(30.0, 3.0 * (attempt + 1)))
                continue
            if 500 <= exc.code < 600 and attempt < retries:
                # 5xx 常常是临时错误，可以再试一次。
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                # 网络抖动或服务端脏响应时，给一次轻量重试。
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"request failed: {last_error}")


def build_openalex_paper(item: dict[str, Any]) -> Paper:
    """Map one OpenAlex work record to the shared Paper model."""
    authors = []
    for authorship in item.get("authorships", []) or []:
        author_name = (((authorship or {}).get("author") or {}).get("display_name") or "").strip()
        if author_name:
            authors.append(Author(author_name))
    primary_location = item.get("primary_location") or {}
    landing_page = primary_location.get("landing_page_url") or item.get("id") or ""
    pdf_url = ((primary_location.get("pdf_url") or "") if isinstance(primary_location, dict) else "")
    return Paper(
        title=str(item.get("title", "")).strip(),
        authors=authors,
        abstract="",
        year=item.get("publication_year"),
        venue=str(((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""),
        doi=str(item.get("doi", "") or ""),
        url=str(landing_page),
        pdf_url=str(pdf_url),
        openalex_id=str(item.get("id", "") or ""),
        source="openalex",
        raw=item,
    )


def search_openalex(query: str, limit: int, year_min: int | None) -> list[Paper]:
    """Search OpenAlex as the default broad-coverage source."""
    params = {
        "search": query,
        "per-page": str(limit),
        "sort": "cited_by_count:desc",
    }
    if year_min is not None:
        params["filter"] = f"from_publication_date:{year_min}-01-01"
    url = "https://api.openalex.org/works?" + parse.urlencode(params)
    payload = request_json(url, source="openalex")
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    return [build_openalex_paper(item) for item in rows if isinstance(item, dict)]


def build_semantic_scholar_paper(item: dict[str, Any]) -> Paper:
    """Map Semantic Scholar JSON into the shared Paper model."""
    authors = [Author(str(author.get("name", "")).strip()) for author in item.get("authors", []) if author.get("name")]
    external_ids = item.get("externalIds") or {}
    url = str(item.get("url", "") or "")
    return Paper(
        title=str(item.get("title", "")).strip(),
        authors=authors,
        abstract=str(item.get("abstract", "") or ""),
        year=item.get("year"),
        venue=str(item.get("venue", "") or ""),
        doi=str(external_ids.get("DOI", "") or item.get("doi", "") or ""),
        url=url,
        semantic_scholar_id=str(item.get("paperId", "") or ""),
        arxiv_id=str(external_ids.get("ArXiv", "") or ""),
        source="semantic_scholar",
        raw=item,
    )


def search_semantic_scholar(query: str, limit: int, year_min: int | None, api_key: str | None) -> list[Paper]:
    """Search Semantic Scholar when the caller wants richer abstract metadata."""
    params = {
        "query": query,
        "limit": str(limit),
        "fields": "title,authors,abstract,year,venue,url,externalIds,paperId,doi",
    }
    if year_min is not None:
        params["year"] = f"{year_min}-"
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + parse.urlencode(params)
    payload = request_json(url, headers=headers, source="semantic_scholar")
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [build_semantic_scholar_paper(item) for item in rows if isinstance(item, dict)]


def build_arxiv_paper(entry_xml: str) -> Paper:
    """Extract a few useful fields from one arXiv Atom entry."""
    title_match = re.search(r"<title>(.*?)</title>", entry_xml, flags=re.S)
    summary_match = re.search(r"<summary>(.*?)</summary>", entry_xml, flags=re.S)
    id_match = re.search(r"<id>https?://arxiv.org/abs/(.*?)</id>", entry_xml)
    published_match = re.search(r"<published>(\d{4})-\d{2}-\d{2}</published>", entry_xml)
    author_matches = re.findall(r"<name>(.*?)</name>", entry_xml)
    pdf_match = re.search(r'href="(https?://arxiv.org/pdf/[^"]+)"', entry_xml)
    authors = [Author(re.sub(r"\s+", " ", name).strip()) for name in author_matches if name.strip()]
    title = re.sub(r"\s+", " ", (title_match.group(1) if title_match else "")).strip()
    abstract = re.sub(r"\s+", " ", (summary_match.group(1) if summary_match else "")).strip()
    arxiv_id = (id_match.group(1) if id_match else "").strip()
    year = int(published_match.group(1)) if published_match else None
    pdf_url = (pdf_match.group(1) if pdf_match else "").strip()
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        venue="arXiv",
        url=url,
        pdf_url=pdf_url,
        arxiv_id=arxiv_id,
        source="arxiv",
        raw={"entry_xml": entry_xml},
    )


def search_arxiv(query: str, limit: int, year_min: int | None) -> list[Paper]:
    """Search arXiv using the Atom API and a lightweight XML parser."""
    params = {
        "search_query": "all:" + query,
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + parse.urlencode(params)
    if breaker_is_open("arxiv"):
        raise RuntimeError("arxiv circuit breaker is open")
    wait_for_rate_limit("arxiv")
    with request.urlopen(url, timeout=20) as response:
        xml_text = response.read().decode("utf-8", errors="replace")
    breaker_record_success("arxiv")
    entries = re.findall(r"<entry>(.*?)</entry>", xml_text, flags=re.S)
    papers = [build_arxiv_paper(entry) for entry in entries]
    if year_min is not None:
        papers = [paper for paper in papers if paper.year is None or paper.year >= year_min]
    return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Merge duplicates with a DOI-first heuristic copied from the repo."""
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        doi_key = f"doi:{paper.doi.lower()}" if paper.doi else ""
        arxiv_key = f"arxiv:{paper.arxiv_id.lower()}" if paper.arxiv_id else ""
        title_key = f"title:{normalize_text(paper.title)}"
        key = doi_key or arxiv_key or title_key
        if key and key not in seen:
            seen.add(key)
            unique.append(paper)
    return unique


def search_one_source(query: str, source: str, config: SearchConfig) -> list[Paper]:
    """Search one source and fall back to cache when the online request fails."""
    # 先读取缓存，但只有在线请求失败时才回退使用。
    cached = cache_get(query, source, config.limit_per_query, config.year_min)
    try:
        if source == "openalex":
            papers = search_openalex(query, config.limit_per_query, config.year_min)
        elif source == "semantic_scholar":
            papers = search_semantic_scholar(
                query,
                config.limit_per_query,
                config.year_min,
                config.semantic_scholar_api_key,
            )
        elif source == "arxiv":
            papers = search_arxiv(query, config.limit_per_query, config.year_min)
        else:
            raise ValueError(f"Unsupported source: {source}")
        if papers:
            # 只有在线请求成功并拿到结果时才刷新缓存。
            cache_put(query, source, config.limit_per_query, config.year_min, papers)
        return papers
    except Exception:
        # 任意源失败都不让整个检索崩掉；有缓存就优先保活。
        if cached:
            return cached
        raise


def search_papers_multi_query(queries: list[str], *, topic: str, config: SearchConfig) -> list[Paper]:
    """Search all sources for all expanded queries and return deduplicated results."""
    all_papers: list[Paper] = []
    # 先扩展查询词，保证短 query / survey / benchmark 等变体能一起召回。
    for query in expand_queries(queries, topic):
        for source in config.sources:
            try:
                all_papers.extend(search_one_source(query, source, config))
            except Exception:
                # 单个源失败时继续尝试其他源，保证“稳定检索”优先。
                continue
    # 最后统一去重并做一个轻量排序，方便下游直接消费。
    ranked = sorted(
        deduplicate_papers(all_papers),
        key=lambda paper: (paper.year or 0, len(paper.abstract), len(paper.title)),
        reverse=True,
    )
    return ranked


def papers_to_jsonl(papers: list[Paper]) -> str:
    """Render search output in the same shape most agent pipelines like to persist."""
    return "\n".join(json.dumps(paper.to_dict(), ensure_ascii=False) for paper in papers) + ("\n" if papers else "")


def papers_to_bibtex(papers: list[Paper]) -> str:
    """Render BibTeX so downstream writing or citation review can reuse the results."""
    return "\n".join(paper.to_bibtex().rstrip() for paper in papers) + ("\n" if papers else "")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a tiny CLI so the module can run without another framework."""
    parser = argparse.ArgumentParser(description="Stable multi-source literature search")
    parser.add_argument("--topic", required=True, help="Research topic or task description")
    parser.add_argument("--query", action="append", default=[], help="Optional seed query; repeatable")
    parser.add_argument("--source", action="append", default=["openalex", "semantic_scholar", "arxiv"])
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--limit-per-query", type=int, default=10)
    parser.add_argument("--s2-api-key", default=None)
    parser.add_argument("--jsonl-out", default="")
    parser.add_argument("--bib-out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for quick local experiments and migration tests."""
    args = build_arg_parser().parse_args(argv)
    config = SearchConfig(
        sources=args.source,
        limit_per_query=args.limit_per_query,
        year_min=args.year_min,
        semantic_scholar_api_key=args.s2_api_key,
    )
    papers = search_papers_multi_query(args.query, topic=args.topic, config=config)
    jsonl_text = papers_to_jsonl(papers)
    bib_text = papers_to_bibtex(papers)
    if args.jsonl_out:
        Path(args.jsonl_out).write_text(jsonl_text, encoding="utf-8")
    if args.bib_out:
        Path(args.bib_out).write_text(bib_text, encoding="utf-8")
    if not args.jsonl_out:
        print(jsonl_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
