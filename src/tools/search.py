"""SearcherAgent 的底层检索工具。

这个文件做三件核心事情：
1) 把用户问题扩展成多条查询（提高召回率）；
2) 并行风格地遍历多个来源并做重试（提高稳定性）；
3) 对论文结果去重并产出统一结构（便于后续落盘和引用）。

注意：这里的目标不是“完美无错”，而是“把错误结构化带回上层给 LLM 决策”。
也就是说，遇到网络/解析异常时，我们尽量不硬崩，而是把错误细节整理出来。
"""

from __future__ import annotations

import re
import time
from typing import Callable


def normalize_title(title: str) -> str:
    # 去掉标点、压缩空白，得到一个稳定的 title key，供去重使用。
    text = (title or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sanitize_token(text: str, *, fallback: str) -> str:
    # 文件名安全化：把不适合当文件名的字符替换为下划线。
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", (text or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def paper_identity(paper: dict) -> tuple[str, str]:
    # 去重优先级：DOI > arXiv ID > 归一化标题。
    # 这样做是因为 DOI/arXiv ID 的唯一性更强，标题最容易撞车。
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return "doi", doi

    arxiv_id = (paper.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return "arxiv", arxiv_id

    title_key = normalize_title(str(paper.get("title") or ""))
    return "title", title_key


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[\.!?。！？])\s+", (text or "").strip())
    return [item.strip() for item in raw if item and item.strip()]


def extract_candidate_claims(paper: dict, *, max_claims: int = 6) -> list[str]:
    """Heuristically extract candidate claims from abstract/body snippets."""
    text_parts = [
        str(paper.get("abstract") or ""),
        str(paper.get("full_text") or ""),
        str(paper.get("summary") or ""),
    ]
    text = "\n".join(part for part in text_parts if part.strip())
    if not text.strip():
        return []

    claim_keywords = (
        "theorem", "lemma", "proposition", "corollary", "bound", "implies", "show", "prove", "result"
    )
    claims: list[str] = []
    for sent in _split_sentences(text):
        lowered = sent.lower()
        if any(keyword in lowered for keyword in claim_keywords):
            claims.append(sent)
        if len(claims) >= max_claims:
            break

    # 兜底策略：如果关键词规则没抓到“像结论的话”，
    # 就按句子长度取前几条，至少给上游一个“可核对候选”。
    if not claims:
        sentences = sorted(_split_sentences(text), key=len, reverse=True)
        claims = sentences[:max_claims]

    return claims


def _merge_paper_records(base: dict, incoming: dict) -> dict:
    # 合并策略：保守保留已有字段，优先吸收“更完整”的新信息。
    merged = dict(base)
    merged_sources = list(dict.fromkeys((base.get("source_list") or [base.get("source")]) + (incoming.get("source_list") or [incoming.get("source")])))
    merged["source_list"] = [item for item in merged_sources if item]

    # Prefer richer fields from incoming record.
    for key in ("title", "doi", "arxiv_id", "url"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming.get(key)

    base_abstract = str(merged.get("abstract") or "")
    incoming_abstract = str(incoming.get("abstract") or "")
    if len(incoming_abstract) > len(base_abstract):
        merged["abstract"] = incoming_abstract

    base_full = str(merged.get("full_text") or "")
    incoming_full = str(incoming.get("full_text") or "")
    if len(incoming_full) > len(base_full):
        merged["full_text"] = incoming_full

    return merged


def dedup_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate papers with DOI > arXiv ID > normalized title priority."""
    seen: dict[tuple[str, str], dict] = {}
    for paper in papers:
        key = paper_identity(paper)
        if not key[1]:
            # 身份键为空说明记录质量太差（既没 DOI/arXiv，也没可用标题）。
            # 这类记录继续留着只会污染后续链路，所以直接丢弃。
            continue

        normalized = dict(paper)
        if "source_list" not in normalized:
            normalized["source_list"] = [normalized.get("source")] if normalized.get("source") else []

        if key in seen:
            seen[key] = _merge_paper_records(seen[key], normalized)
            continue
        seen[key] = normalized

    return list(seen.values())


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

    # 轻量、可解释的“确定性扩展”：不依赖 LLM，也不引入随机性。
    # 目的是给搜索引擎更多可命中的语义变体（proof/theorem/survey/arxiv）。
    expanded: list[str] = []
    suffixes = ["proof", "theorem", "survey", "arxiv"]
    for item in out:
        expanded.append(item)
        lowered = item.lower()
        for suffix in suffixes:
            candidate = f"{item} {suffix}"
            candidate_key = candidate.lower()
            if suffix in lowered or candidate_key in seen:
                continue
            seen.add(candidate_key)
            expanded.append(candidate)
    return expanded


def _format_source_error(
    *,
    source: str,
    query: str,
    attempt: int,
    max_attempts: int,
    exc: BaseException,
) -> dict:
    """把来源异常压成统一、可给 LLM 读取的结构化信息。"""
    return {
        "source": source,
        "query": query,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def multi_source_search_with_diagnostics(
    queries: list[str],
    source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None,
    limit_per_query: int = 10,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 0.2,
) -> dict:
    """聚合多来源检索，并返回“结果 + 错误诊断”。

    返回结构：
    {
      "results": [...],
      "errors": [...],              # 最终仍失败的查询
      "recovered_errors": [...],    # 先失败后成功（可观察稳定性问题）
      "stats": {...}
    }

    设计动机：
    - 以前遇错会静默降级为空列表，LLM 看不到失败细节；
    - 现在把失败上下文完整带回去，让 LLM 决定“换 query / 换策略 / 继续推理”。
    """
    if not source_handlers:
        return {
            "results": [],
            "errors": [
                {
                    "source": "N/A",
                    "query": "",
                    "attempt": 0,
                    "max_attempts": max(1, retry_attempts),
                    "error_type": "NO_SOURCE_HANDLERS",
                    "error_message": "No search source handlers configured.",
                }
            ],
            "recovered_errors": [],
            "stats": {
                "queries": len(queries or []),
                "sources": 0,
                "attempts": 0,
                "failed_queries": 1,
                "recovered_queries": 0,
            },
        }

    results: list[dict] = []
    errors: list[dict] = []
    recovered_errors: list[dict] = []
    total_attempts = 0
    normalized_max_attempts = max(1, retry_attempts)

    # 这层循环是“来源 x 查询”笛卡尔积：
    # 每个来源都尝试每条 query，避免单来源盲区导致漏检。
    for source_name, handler in source_handlers.items():
        for query in queries:
            items: list[dict] = []
            transient_errors: list[dict] = []

            for attempt in range(1, normalized_max_attempts + 1):
                total_attempts += 1
                try:
                    items = handler(query, limit_per_query) or []
                    break
                except BaseException as exc:
                    # SystemExit / GeneratorExit 属于解释器级终止，不应该吞掉。
                    if isinstance(exc, (SystemExit, GeneratorExit)):
                        raise

                    transient_errors.append(
                        _format_source_error(
                            source=source_name,
                            query=query,
                            attempt=attempt,
                            max_attempts=normalized_max_attempts,
                            exc=exc,
                        )
                    )

                    if attempt >= normalized_max_attempts:
                        items = []
                    else:
                        time.sleep(max(0.0, retry_backoff_seconds))

            # 先失败后成功：记录为 recovered，便于后续做来源稳定性分析。
            if transient_errors and items:
                recovered_errors.extend(transient_errors)

            # 最终失败：记录为 errors，交给上层 LLM 做下一步策略决策。
            if transient_errors and not items:
                errors.append(transient_errors[-1])

            for item in items:
                record = dict(item)
                record.setdefault("source", source_name)
                record.setdefault("query", query)
                results.append(record)

    return {
        "results": results,
        "errors": errors,
        "recovered_errors": recovered_errors,
        "stats": {
            "queries": len(queries or []),
            "sources": len(source_handlers),
            "attempts": total_attempts,
            "failed_queries": len(errors),
            "recovered_queries": len(recovered_errors),
        },
    }


def multi_source_search(
    queries: list[str],
    source_handlers: dict[str, Callable[[str, int], list[dict]]] | None = None,
    limit_per_query: int = 10,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 0.2,
) -> list[dict]:
    """兼容旧调用：仅返回结果列表，不暴露诊断字段。"""
    diagnostics = multi_source_search_with_diagnostics(
        queries,
        source_handlers=source_handlers,
        limit_per_query=limit_per_query,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return diagnostics["results"]


def build_paper_filename(paper: dict) -> str:
    # 文件名优先用强身份（DOI/arXiv），找不到再退化到标题。
    doi = (paper.get("doi") or "").strip()
    if doi:
        return f"doi_{_sanitize_token(doi, fallback='paper')}.md"

    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"arXiv_{_sanitize_token(arxiv_id, fallback='paper')}.md"

    title = normalize_title(str(paper.get("title") or ""))
    return f"title_{_sanitize_token(title[:80], fallback='paper')}.md"


def build_paper_markdown(paper: dict) -> str:
    # 这里把“结构化检索记录”转成可审计的 markdown：
    # - Layer2 放正文片段
    # - Candidate Claims 放候选主张
    # - Layer3 放来源与元信息
    title = str(paper.get("title") or "Unknown Title")
    abstract = str(paper.get("abstract") or "")
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        author_line = "; ".join(str(a) for a in authors)
    else:
        author_line = str(authors)

    summary = abstract.strip()[:240] if abstract else f"Paper summary for: {title}"
    layer2_source = str(paper.get("full_text") or "").strip() or abstract.strip()
    layer2 = layer2_source or "No extracted body available."
    candidate_claims = extract_candidate_claims(paper)

    source_list = paper.get("source_list") or []
    if isinstance(source_list, list):
        source_list_text = "; ".join(str(item) for item in source_list if str(item).strip())
    else:
        source_list_text = str(source_list)

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
        "### Candidate Claims",
    ]

    if candidate_claims:
        lines.extend([f"- {claim}" for claim in candidate_claims])
    else:
        lines.append("- NONE")

    lines.extend([
        "",
        "## Layer3-Source",
        f"source: {paper.get('source') or ''}",
        f"source_list: {source_list_text}",
        f"url: {paper.get('url') or ''}",
        f"authors: {author_line}",
    ])
    return "\n".join(lines)
