"""Default online source handlers used by SearcherAgent."""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

_USER_AGENT = "Aletheia-Searcher/1.0"
_OPENALEX_WORKS_API = "https://api.openalex.org/works"
_ARXIV_QUERY_API = "https://export.arxiv.org/api/query"
_SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"

_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _http_get(url: str, *, timeout: int = 20, headers: dict[str, str] | None = None) -> bytes:
    # 某些学术源在企业/校园网络下会出现 SSL 链异常，
    # 这里保留“宽松 SSL 回退”作为兜底，优先保证可用性。
    def _make_lenient_ssl_context() -> ssl.SSLContext:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    request_headers = {"User-Agent": _USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except ssl.SSLError:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=_make_lenient_ssl_context(),
        ) as response:
            return response.read()
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLError):
            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=_make_lenient_ssl_context(),
            ) as response:
                return response.read()
        raise


def _decode_openalex_abstract(inverted_index: dict | None) -> str:
    # OpenAlex 的摘要是倒排索引格式：token -> positions。
    # 这里按位置重排回自然文本。
    if not isinstance(inverted_index, dict):
        return ""

    tokens: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                tokens.append((int(pos), str(token)))
            except (TypeError, ValueError):
                continue

    if not tokens:
        return ""

    tokens.sort(key=lambda item: item[0])
    return " ".join(word for _, word in tokens)


def _normalize_doi(raw_doi: str | None) -> str:
    value = (raw_doi or "").strip()
    if value.startswith("https://doi.org/"):
        return value.replace("https://doi.org/", "", 1)
    if value.startswith("http://doi.org/"):
        return value.replace("http://doi.org/", "", 1)
    return value


def search_openalex(query: str, limit: int) -> list[dict]:
    # 标准化查询并限制上限，避免 API 负载过高。
    q = (query or "").strip()
    if not q:
        return []

    capped_limit = max(1, min(int(limit or 10), 25))
    params = urllib.parse.urlencode(
        {
            "search": q,
            "per-page": capped_limit,
        }
    )
    payload = json.loads(_http_get(f"{_OPENALEX_WORKS_API}?{params}").decode("utf-8"))

    papers: list[dict] = []
    for item in payload.get("results", []) or []:
        title = str(item.get("display_name") or "").strip()
        if not title:
            continue

        doi = _normalize_doi(((item.get("ids") or {}).get("doi") or ""))
        abstract = _decode_openalex_abstract(item.get("abstract_inverted_index"))

        authors: list[str] = []
        for authorship in item.get("authorships", []) or []:
            author_name = ((authorship or {}).get("author") or {}).get("display_name")
            if author_name:
                authors.append(str(author_name))

        primary_location = item.get("primary_location") or {}
        url = (
            primary_location.get("landing_page_url")
            or primary_location.get("pdf_url")
            or item.get("id")
            or ""
        )

        papers.append(
            {
                "title": title,
                "doi": doi,
                "abstract": abstract,
                "authors": authors,
                "url": str(url),
            }
        )
    return papers


def search_arxiv(query: str, limit: int) -> list[dict]:
    # arXiv 返回 Atom XML，这里统一转换为项目内部 paper 字典结构。
    q = (query or "").strip()
    if not q:
        return []

    capped_limit = max(1, min(int(limit or 10), 25))
    encoded_query = urllib.parse.quote(f"all:{q}")
    url = (
        f"{_ARXIV_QUERY_API}?search_query={encoded_query}"
        f"&start=0&max_results={capped_limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    xml_bytes = _http_get(url)
    root = ET.fromstring(xml_bytes)

    papers: list[dict] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_element = entry.find("atom:title", _ATOM_NS)
        title = (title_element.text or "").replace("\n", " ").strip() if title_element is not None else ""
        if not title:
            continue

        summary_element = entry.find("atom:summary", _ATOM_NS)
        abstract = (summary_element.text or "").replace("\n", " ").strip() if summary_element is not None else ""

        id_element = entry.find("atom:id", _ATOM_NS)
        id_url = (id_element.text or "").strip() if id_element is not None else ""
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

        doi_element = entry.find("arxiv:doi", _ATOM_NS)
        doi = (doi_element.text or "").strip() if doi_element is not None else ""

        authors: list[str] = []
        for author in entry.findall("atom:author", _ATOM_NS):
            name_element = author.find("atom:name", _ATOM_NS)
            name = (name_element.text or "").strip() if name_element is not None else ""
            if name:
                authors.append(name)

        papers.append(
            {
                "title": title,
                "arxiv_id": arxiv_id,
                "doi": _normalize_doi(doi),
                "abstract": abstract,
                "authors": authors,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else id_url,
            }
        )
    return papers


def search_semantic_scholar(query: str, limit: int, *, api_key: str | None = None) -> list[dict]:
    # Semantic Scholar 支持可选 API key：
    # - 有 key：更稳定
    # - 无 key：仍可尝试匿名访问（受限）
    q = (query or "").strip()
    if not q:
        return []

    capped_limit = max(1, min(int(limit or 10), 20))
    params = urllib.parse.urlencode(
        {
            "query": q,
            "limit": capped_limit,
            "fields": "title,abstract,authors,url,externalIds",
        }
    )
    headers = {}
    key = (api_key or "").strip()
    if key:
        headers["x-api-key"] = key

    payload = json.loads(_http_get(f"{_SEMANTIC_SCHOLAR_SEARCH_API}?{params}", headers=headers).decode("utf-8"))

    papers: list[dict] = []
    for item in payload.get("data", []) or []:
        title = str(item.get("title") or "").strip()
        if not title:
            continue

        authors = [str((author or {}).get("name")) for author in (item.get("authors") or []) if (author or {}).get("name")]
        external_ids = item.get("externalIds") or {}

        papers.append(
            {
                "title": title,
                "abstract": str(item.get("abstract") or "").strip(),
                "doi": _normalize_doi(external_ids.get("DOI") or ""),
                "arxiv_id": str(external_ids.get("ArXiv") or "").strip(),
                "authors": authors,
                "url": str(item.get("url") or ""),
            }
        )
    return papers


def build_default_source_handlers(config: dict | None = None) -> dict[str, Callable[[str, int], list[dict]]]:
    """Build production source handlers for SearcherAgent.

    Retrieval source order can be configured by `retrieval.sources` in settings.yaml.
    """
    settings = config if isinstance(config, dict) else {}
    retrieval_cfg = settings.get("retrieval") if isinstance(settings.get("retrieval"), dict) else {}
    tools_cfg = settings.get("tools") if isinstance(settings.get("tools"), dict) else {}

    configured_sources = retrieval_cfg.get("sources")
    if isinstance(configured_sources, list) and configured_sources:
        sources = [str(item).strip().lower() for item in configured_sources if str(item).strip()]
    else:
        sources = ["openalex", "arxiv"]

    semantic_scholar_api_key = str(tools_cfg.get("semantic_scholar_api_key") or "").strip() or None

    # 返回 source_name -> handler 的映射；SearcherAgent 会按该映射遍历调用。
    handlers: dict[str, Callable[[str, int], list[dict]]] = {}
    for source in sources:
        if source == "openalex":
            handlers[source] = search_openalex
        elif source == "arxiv":
            handlers[source] = search_arxiv
        elif source == "semantic_scholar":
            handlers[source] = lambda query, limit, api_key=semantic_scholar_api_key: search_semantic_scholar(
                query,
                limit,
                api_key=api_key,
            )
    return handlers
