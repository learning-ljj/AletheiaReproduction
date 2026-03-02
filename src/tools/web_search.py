"""学术搜索与 LaTeX 源码提取工具（arXiv API + 标准库）。"""

import gzip
import io
import logging
import ssl
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# arXiv API 配置
_ARXIV_QUERY_URL = "http://export.arxiv.org/api/query"
_ARXIV_EPRINT_URL = "https://export.arxiv.org/e-print"
_USER_AGENT = "Mozilla/5.0 (Aletheia-Agent)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_FETCH_RETRIES = 3  # Bug #2 修复：网络请求失败时最多重试次数


def _make_lenient_ssl_ctx() -> ssl.SSLContext:
    """创建宽松 SSL 上下文，用于 SSL 握手失败的降级重试。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _urlopen_with_retry(req: urllib.request.Request, timeout: int) -> bytes:
    """带重试与 SSL 降级的 urlopen 封装。

    重试策略：
      1. 首次使用默认 SSL（标准安全）；
      2. 若 SSL 错误，降级到宽松 SSL 上下文重试；
      3. 其他网络错误：最多 _MAX_FETCH_RETRIES 次指数退避。
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_FETCH_RETRIES):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))  # 1 s → 2 s 指数退避
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except ssl.SSLError as e:
            logger.warning("SSL error on attempt %d/%d: %s — retrying with lenient SSL context.",
                           attempt + 1, _MAX_FETCH_RETRIES, e)
            try:
                with urllib.request.urlopen(req, timeout=timeout,
                                            context=_make_lenient_ssl_ctx()) as resp:
                    return resp.read()
            except Exception as e2:
                last_error = e2
                logger.warning("Lenient SSL retry also failed: %s", e2)
        except Exception as e:
            last_error = e
            logger.warning("Network error on attempt %d/%d: %s", attempt + 1, _MAX_FETCH_RETRIES, e)
    raise last_error or RuntimeError("_urlopen_with_retry: unexpected failure")


def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """通过 arXiv API 搜索学术论文（两步验证 Step 1）。

    返回 [{"arxiv_id": ..., "title": ..., "authors": ..., "published": ..., "snippet": ...}, ...]
    搜索失败或无结果返回空列表。
    """
    encoded_query = urllib.parse.quote(f"all:{query}")
    url = (
        f"{_ARXIV_QUERY_URL}?search_query={encoded_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    try:
        time.sleep(1)  # 遵循 arXiv API 限流规范
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        xml_data = _urlopen_with_retry(req, timeout=15)

        root = ET.fromstring(xml_data)
        results: list[dict] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            # 提取 arXiv ID (e.g. "http://arxiv.org/abs/2501.12345v1" → "2501.12345v1")
            id_url = entry.find("atom:id", _ATOM_NS).text
            arxiv_id = id_url.split("/abs/")[-1]

            raw_title = entry.find("atom:title", _ATOM_NS)
            title = (raw_title.text or "").replace("\n", " ").strip() if raw_title is not None else ""
            raw_summary = entry.find("atom:summary", _ATOM_NS)
            summary = (raw_summary.text or "").replace("\n", " ").strip() if raw_summary is not None else ""
            raw_pub = entry.find("atom:published", _ATOM_NS)
            published = raw_pub.text if raw_pub is not None else ""

            authors = [
                name_el.text
                for a in entry.findall("atom:author", _ATOM_NS)
                if (name_el := a.find("atom:name", _ATOM_NS)) is not None
                and name_el.text
            ]

            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": ", ".join(authors),
                "published": published,
                "snippet": summary,
            })

        return results

    except Exception as e:
        logger.warning("search_arxiv 请求失败: %s", e)
        return []


def _extract_key_sections(latex: str, max_chars: int = 4000) -> str:
    """从 LaTeX 源码中提取摘要和主要结果/定理章节，避免全文污染 LLM 上下文。

    提取顺序（贪心，满足 max_chars 后截止）：
      1. \\begin{abstract}...\\end{abstract}
      2. 标题匹配关键词的 \\section{...}：
         main results / theorem(s) / key findings / conclusion
    若以上均未找到，回退到原文前 max_chars 字符。
    """
    import re as _re

    parts: list[str] = []

    # --- 1. 提取 abstract ---
    abstract_match = _re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex, _re.DOTALL
    )
    if abstract_match:
        parts.append("[ABSTRACT]\n" + abstract_match.group(1).strip())

    # --- 2. 提取关键章节 ---
    _KEY_SECTION_RE = _re.compile(
        r"\\(?:section|subsection)\*?\{([^}]*(?:main\s+result|theorem|key\s+finding|"
        r"overview|conclusion)[^}]*)\}"
        r"(.*?)(?=\\(?:section|subsection)\*?\{|\\end\{document\}|\Z)",
        _re.IGNORECASE | _re.DOTALL,
    )
    for m in _KEY_SECTION_RE.finditer(latex):
        title = m.group(1).strip()
        body = m.group(2).strip()
        # 只取每个章节前 1000 字符，防止单章节过长
        snippet = f"[SECTION: {title}]\n{body[:1000]}"
        parts.append(snippet)
        if sum(len(p) for p in parts) >= max_chars:
            break

    if parts:
        result = "\n\n".join(parts)
        return result[:max_chars]

    # --- 回退：直接截断 ---
    return latex[:max_chars]


def read_arxiv_latex(arxiv_id: str, max_chars: int = 4000) -> str:
    """根据 arXiv ID 下载 e-print 源文件并提取关键章节（两步验证 Step 2）。

    仅返回摘要与主要结果/定理章节（≤ max_chars），避免全文 LaTeX 污染 LLM 上下文。
    支持 tar.gz（多文件）和纯 gzip（单文件）两种格式。
    失败返回错误描述字符串。
    """
    url = f"{_ARXIV_EPRINT_URL}/{arxiv_id}"
    try:
        time.sleep(1)  # 遵循 arXiv API 限流规范
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        file_bytes = _urlopen_with_retry(req, timeout=30)

        stream = io.BytesIO(file_bytes)
        tex_content: list[str] = []

        # 尝试 1: tar.gz 格式（多文件论文）
        try:
            with tarfile.open(fileobj=stream, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith((".tex", ".bbl")):
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8", errors="ignore")
                            tex_content.append(f"--- File: {member.name} ---\n{content}")
        except tarfile.ReadError:
            # 尝试 2: 纯 gzip 格式（单文件论文）
            stream.seek(0)
            try:
                with gzip.GzipFile(fileobj=stream) as gz:
                    content = gz.read().decode("utf-8", errors="ignore")
                    tex_content.append(content)
            except Exception:
                return (
                    f"Error: Cannot parse source format for {arxiv_id}. "
                    "It might be a PDF-only submission."
                )

        if not tex_content:
            return f"Error: No .tex files found in the source archive for {arxiv_id}."

        full_text = "\n\n".join(tex_content)
        return _extract_key_sections(full_text, max_chars=max_chars)

    except Exception as e:
        return f"Error fetching LaTeX source for {arxiv_id}: {e}"
