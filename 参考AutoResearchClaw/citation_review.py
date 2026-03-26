"""Low-coupling citation verification module inspired by ResearchClaw.

Core ideas preserved from the original repo:
1. Verify citations in a conservative order: DOI -> OpenAlex -> arXiv -> title search.
2. Cache verification results so repeated runs stay cheap and stable.
3. Keep a rollback-friendly filtered BibTeX output.
4. Allow optional topic relevance scoring without coupling to one LLM client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request
import argparse
import hashlib
import json
import re
import time

from docs.analysis.migration_core.shared_models import CitationResult, CitationStatus, VerificationReport
from docs.analysis.migration_core.stable_literature_search import SearchConfig, normalize_text, search_papers_multi_query


# Keep verification cache outside repo-specific directories.
CACHE_ROOT = Path(".migration_cache") / "citation_verify"

# The full-stage timeout mirrors the original repo's defensive behavior.
DEFAULT_TIMEOUT_SECONDS = 300.0

# Verification cache does not need minute-level freshness.
CACHE_TTL = timedelta(days=7)


@dataclass(slots=True)
class BibEntry:
    """A minimal BibTeX entry representation suitable for verification."""

    key: str
    entry_type: str
    raw_text: str
    fields: dict[str, str]


def utcnow() -> datetime:
    """Centralize current UTC time to keep tests simple."""
    return datetime.now(timezone.utc)


def ensure_cache_dir() -> None:
    """Create the local verification cache directory when needed."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def verification_cache_path(key: str, title: str, doi: str) -> Path:
    """Return a stable file path for one verification attempt."""
    ensure_cache_dir()
    payload = json.dumps({"key": key, "title": title, "doi": doi}, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CACHE_ROOT / f"{digest}.json"


def verification_cache_get(key: str, title: str, doi: str) -> CitationResult | None:
    """Read a cached verification result when it is still fresh."""
    path = verification_cache_path(key, title, doi)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        created_at = datetime.fromisoformat(str(payload.get("created_at", "")))
    except ValueError:
        return None
    if utcnow() - created_at > CACHE_TTL:
        return None
    status_text = str(payload.get("status", CitationStatus.SKIPPED.value))
    try:
        status = CitationStatus(status_text)
    except ValueError:
        status = CitationStatus.SKIPPED
    return CitationResult(
        key=str(payload.get("key", key)),
        title=str(payload.get("title", title)),
        status=status,
        reason=str(payload.get("reason", "")),
        matched_title=str(payload.get("matched_title", "")),
        matched_doi=str(payload.get("matched_doi", "")),
        source=str(payload.get("source", "")),
        score=float(payload.get("score", 0.0)),
    )


def verification_cache_put(result: CitationResult) -> None:
    """Persist one verification result so repeated runs can reuse it."""
    path = verification_cache_path(result.key, result.title, result.matched_doi or "")
    payload = {
        "created_at": utcnow().isoformat(),
        "key": result.key,
        "title": result.title,
        "status": result.status.value,
        "reason": result.reason,
        "matched_title": result.matched_title,
        "matched_doi": result.matched_doi,
        "source": result.source,
        "score": result.score,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_bibtex_entries(bib_text: str) -> list[BibEntry]:
    """Parse BibTeX with a regex-first approach that is easy to migrate."""
    # 先找到每个条目的头部位置，后面再切 raw_text。
    matches = list(re.finditer(r"@(\w+)\s*\{\s*([^,]+),", bib_text))
    entries: list[BibEntry] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(bib_text)
        raw_text = bib_text[start:end].strip()
        fields: dict[str, str] = {}
        for field_match in re.finditer(r"(\w+)\s*=\s*[\{\"](.+?)[\}\"],?\s*$", raw_text, flags=re.M | re.S):
            # 把 title / doi / year 等字段统一压平成简单字符串。
            field_name = field_match.group(1).lower()
            field_value = re.sub(r"\s+", " ", field_match.group(2)).strip()
            fields[field_name] = field_value
        entries.append(
            BibEntry(
                key=match.group(2).strip(),
                entry_type=match.group(1).strip().lower(),
                raw_text=raw_text,
                fields=fields,
            )
        )
    return entries


def title_similarity(left: str, right: str) -> float:
    """Compute a light-weight similarity score without third-party libraries."""
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(len(left_tokens), len(right_tokens))


def request_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    """Fetch JSON with small retry logic for transient network errors."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            req = request.Request(url, headers=headers or {})
            with request.urlopen(req, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                return None
            if attempt < 2 and (exc.code == 429 or 500 <= exc.code < 600):
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"request failed: {last_error}")


def verify_by_doi(doi: str, title: str) -> CitationResult | None:
    """Verify by DOI first because it is the most precise identifier."""
    normalized_doi = doi.strip().lower()
    if not normalized_doi:
        return None
    safe_doi = parse.quote(normalized_doi, safe="")
    for source_name, url in [
        ("crossref", f"https://api.crossref.org/works/{safe_doi}"),
        ("datacite", f"https://api.datacite.org/dois/{safe_doi}"),
    ]:
        payload = request_json(url)
        if not payload:
            continue
        if source_name == "crossref":
            message = payload.get("message", {}) if isinstance(payload, dict) else {}
            matched_title = " ".join(message.get("title", []) or [])
            matched_doi = str(message.get("DOI", "") or normalized_doi)
        else:
            attributes = (((payload or {}).get("data") or {}).get("attributes") or {})
            titles = attributes.get("titles", []) or []
            matched_title = " ".join(str(item.get("title", "")) for item in titles if isinstance(item, dict))
            matched_doi = str(attributes.get("doi", "") or normalized_doi)
        score = title_similarity(title, matched_title)
        if score >= 0.6:
            return CitationResult(
                key="",
                title=title,
                status=CitationStatus.VERIFIED,
                reason="DOI matched authoritative registry",
                matched_title=matched_title,
                matched_doi=matched_doi,
                source=source_name,
                score=score,
            )
    return None


def verify_by_openalex(title: str) -> CitationResult | None:
    """Verify by OpenAlex title search because coverage is broad and stable."""
    if not title.strip():
        return None
    params = parse.urlencode({"search": title, "per-page": "5"})
    payload = request_json("https://api.openalex.org/works?" + params)
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    best_title = ""
    best_doi = ""
    best_score = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_title = str(row.get("title", "")).strip()
        score = title_similarity(title, candidate_title)
        if score > best_score:
            best_score = score
            best_title = candidate_title
            best_doi = str(row.get("doi", "") or "")
    if best_score >= 0.8:
        return CitationResult(
            key="",
            title=title,
            status=CitationStatus.VERIFIED,
            reason="High title similarity in OpenAlex",
            matched_title=best_title,
            matched_doi=best_doi,
            source="openalex",
            score=best_score,
        )
    if best_score >= 0.55:
        return CitationResult(
            key="",
            title=title,
            status=CitationStatus.SUSPICIOUS,
            reason="Medium title similarity in OpenAlex",
            matched_title=best_title,
            matched_doi=best_doi,
            source="openalex",
            score=best_score,
        )
    return None


def verify_by_arxiv_id(arxiv_id: str, title: str) -> CitationResult | None:
    """Verify by arXiv id when the BibTeX entry contains one."""
    if not arxiv_id.strip():
        return None
    params = parse.urlencode({"id_list": arxiv_id.strip()})
    url = "http://export.arxiv.org/api/query?" + params
    with request.urlopen(url, timeout=20) as response:
        xml_text = response.read().decode("utf-8", errors="replace")
    title_match = re.search(r"<entry>.*?<title>(.*?)</title>", xml_text, flags=re.S)
    matched_title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    score = title_similarity(title, matched_title)
    if score >= 0.7:
        return CitationResult(
            key="",
            title=title,
            status=CitationStatus.VERIFIED,
            reason="arXiv identifier resolved to similar title",
            matched_title=matched_title,
            matched_doi="",
            source="arxiv",
            score=score,
        )
    return None


def verify_by_title_search(title: str, semantic_scholar_api_key: str | None) -> CitationResult | None:
    """Final fallback: search by title using the standalone search module."""
    config = SearchConfig(
        sources=["semantic_scholar", "openalex", "arxiv"],
        limit_per_query=3,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )
    papers = search_papers_multi_query([title], topic=title, config=config)
    if not papers:
        return None
    best = max(papers, key=lambda paper: title_similarity(title, paper.title))
    score = title_similarity(title, best.title)
    if score >= 0.8:
        status = CitationStatus.VERIFIED
        reason = "High similarity in title-search fallback"
    elif score >= 0.55:
        status = CitationStatus.SUSPICIOUS
        reason = "Medium similarity in title-search fallback"
    else:
        return None
    return CitationResult(
        key="",
        title=title,
        status=status,
        reason=reason,
        matched_title=best.title,
        matched_doi=best.doi,
        source=best.source,
        score=score,
    )


def verify_one_entry(entry: BibEntry, semantic_scholar_api_key: str | None) -> CitationResult:
    """Verify one BibTeX entry using the repo's preferred source order."""
    title = entry.fields.get("title", "").strip()
    doi = entry.fields.get("doi", "").strip()
    arxiv_id = entry.fields.get("eprint", "").strip() or entry.fields.get("arxiv", "").strip()
    # 先查缓存，避免每次运行都重复访问外部 API。
    cached = verification_cache_get(entry.key, title, doi)
    if cached:
        cached.key = entry.key
        cached.title = title
        return cached
    result = None
    if doi:
        # DOI 最精确，所以永远优先。
        result = verify_by_doi(doi, title)
    if result is None:
        # DOI 不可用时，OpenAlex 是更稳的标题检索兜底。
        result = verify_by_openalex(title)
    if result is None and arxiv_id:
        # 有 arXiv id 就走精确匹配，不依赖模糊标题。
        result = verify_by_arxiv_id(arxiv_id, title)
    if result is None:
        # 最后才退到通用标题搜索，成本最高、精度也最不稳定。
        result = verify_by_title_search(title, semantic_scholar_api_key)
    if result is None:
        # 所有外部源都证实不了时，按幻觉引用处理。
        result = CitationResult(
            key=entry.key,
            title=title,
            status=CitationStatus.HALLUCINATED,
            reason="No external source could verify this citation",
            source="",
        )
    else:
        # 复用外部核验结果，但把当前 BibTeX key 补回去。
        result.key = entry.key
        result.title = title
    verification_cache_put(result)
    return result


def verify_citations(
    bib_text: str,
    *,
    semantic_scholar_api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    topic: str | None = None,
    relevance_scorer: Callable[[str, list[CitationResult]], dict[str, float]] | None = None,
) -> VerificationReport:
    """Verify all citations and optionally apply a topic relevance scoring layer."""
    started_at = time.time()
    entries = parse_bibtex_entries(bib_text)
    results: list[CitationResult] = []
    for entry in entries:
        # 全局超时到了之后，不再继续请求外部 API，剩余条目标记 skipped。
        if time.time() - started_at > timeout_seconds:
            results.append(
                CitationResult(
                    key=entry.key,
                    title=entry.fields.get("title", ""),
                    status=CitationStatus.SKIPPED,
                    reason="Verification timed out before this entry was processed",
                )
            )
            continue
        try:
            results.append(verify_one_entry(entry, semantic_scholar_api_key))
        except Exception as exc:
            # 单条失败不影响其他条目，避免一条坏引用拖垮整次核验。
            results.append(
                CitationResult(
                    key=entry.key,
                    title=entry.fields.get("title", ""),
                    status=CitationStatus.SKIPPED,
                    reason=f"Verification failed with error: {exc}",
                )
            )
    report = VerificationReport(results=results)
    if topic and relevance_scorer and report.results:
        # 如果外层 agent 想加“是否和当前主题相关”的复核，就从这里插入。
        scores = relevance_scorer(topic, report.results)
        for item in report.results:
            if item.key in scores:
                item.score = scores[item.key]
                if item.status == CitationStatus.VERIFIED and item.score < 0.2:
                    # 真实性通过但相关性太低时，降成 suspicious，交给人审。
                    item.status = CitationStatus.SUSPICIOUS
                    item.reason = "Verified existence but low topic relevance"
    return report


def filter_verified_bibtex(bib_text: str, report: VerificationReport, include_suspicious: bool = True) -> str:
    """Keep only entries that survived verification."""
    keep_statuses = {CitationStatus.VERIFIED, CitationStatus.SKIPPED}
    if include_suspicious:
        keep_statuses.add(CitationStatus.SUSPICIOUS)
    keep_keys = {item.key for item in report.results if item.status in keep_statuses}
    entries = parse_bibtex_entries(bib_text)
    kept = [entry.raw_text for entry in entries if entry.key in keep_keys]
    return "\n\n".join(kept) + ("\n" if kept else "")


def remove_bibtex_entries(bib_text: str, keys_to_remove: set[str]) -> str:
    """Remove specific BibTeX entries by key."""
    entries = parse_bibtex_entries(bib_text)
    kept = [entry.raw_text for entry in entries if entry.key not in keys_to_remove]
    return "\n\n".join(kept) + ("\n" if kept else "")


def prune_uncited_bib_entries(bib_text: str, paper_markdown: str) -> str:
    """Drop BibTeX entries that are never cited in the paper draft."""
    cited_keys = set(re.findall(r"\[([A-Za-z0-9:_-]+)\]", paper_markdown))
    entries = parse_bibtex_entries(bib_text)
    kept = [entry.raw_text for entry in entries if entry.key in cited_keys]
    return "\n\n".join(kept) + ("\n" if kept else "")


def annotate_paper_hallucinations(paper_markdown: str, report: VerificationReport, low_relevance_keys: set[str] | None = None) -> str:
    """Remove hallucinated or low-relevance cite keys from a markdown draft."""
    low_relevance_keys = low_relevance_keys or set()
    bad_keys = {
        item.key
        for item in report.results
        if item.status == CitationStatus.HALLUCINATED
    } | low_relevance_keys
    updated = paper_markdown
    for key in bad_keys:
        # 这里只做最小改写：直接删掉坏 cite key，正文内容保持不动。
        updated = updated.replace(f"[{key}]", "")
    return re.sub(r"\s{2,}", " ", updated)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a small CLI for standalone verification runs."""
    parser = argparse.ArgumentParser(description="Verify and filter BibTeX citations")
    parser.add_argument("--bib", required=True, help="Path to input references.bib")
    parser.add_argument("--out-bib", default="", help="Optional path to filtered BibTeX output")
    parser.add_argument("--report-json", default="", help="Optional path to JSON report")
    parser.add_argument("--paper-md", default="", help="Optional markdown file for cite cleanup")
    parser.add_argument("--paper-out", default="", help="Optional cleaned markdown output")
    parser.add_argument("--topic", default="", help="Optional topic for relevance scoring")
    parser.add_argument("--s2-api-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint that verifies BibTeX and optionally rewrites markdown."""
    args = build_arg_parser().parse_args(argv)
    bib_text = Path(args.bib).read_text(encoding="utf-8")
    report = verify_citations(
        bib_text,
        semantic_scholar_api_key=args.s2_api_key,
        topic=args.topic or None,
    )
    filtered_bib = filter_verified_bibtex(bib_text, report, include_suspicious=True)
    if args.out_bib:
        Path(args.out_bib).write_text(filtered_bib, encoding="utf-8")
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.paper_md and args.paper_out:
        paper_text = Path(args.paper_md).read_text(encoding="utf-8")
        cleaned = annotate_paper_hallucinations(paper_text, report)
        Path(args.paper_out).write_text(cleaned, encoding="utf-8")
    if not args.report_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
