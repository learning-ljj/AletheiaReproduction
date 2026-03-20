"""Wikipedia 检索工具：MediaWiki Action API + 内容清洗。"""

import html
import logging
import re
import time
import urllib.parse
import urllib.request

from src.tools._http_utils import urlopen_with_retry

logger = logging.getLogger(__name__)

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "Aletheia-Agent/1.0 (Educational Purpose)"
_MAX_CONTENT_CHARS = 8000


# ------------------------------------------------------------------
# MediaWiki API 底层请求
# ------------------------------------------------------------------


def _wiki_request(params: dict) -> dict:
    base = {"format": "json", "formatversion": "2", "redirects": "1"}
    all_params = {**params, **base}
    url = _WIKI_API + "?" + urllib.parse.urlencode(all_params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    import json
    return json.loads(urlopen_with_retry(req, timeout=15).decode("utf-8"))


def _search_pages(term: str, limit: int = 5) -> list[dict]:
    """MediaWiki Action API search。"""
    data = _wiki_request({
        "action": "query", "list": "search",
        "srsearch": term, "srlimit": limit, "srprop": "title|pageid",
    })
    return [
        {"title": item["title"], "pageid": item["pageid"]}
        for item in data.get("query", {}).get("search", [])
    ]


def _fetch_page_html(title: str) -> str | None:
    """获取页面 HTML 原文。"""
    try:
        data = _wiki_request({"action": "parse", "page": title, "prop": "text"})
        if "error" in data:
            return None
        return data.get("parse", {}).get("text", "")
    except Exception as e:
        logger.warning("fetch_page_html failed for %r: %s", title, e)
        return None


# ------------------------------------------------------------------
# 内容清洗（参考 _normalize_external_content 流水线）
# ------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    """HTML → 纯文本清洗流水线。"""
    if not raw:
        return ""

    # Phase 1: 删除 script/style/noscript 块
    text = re.sub(r"<script[\s\S]*?</script>", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<noscript[\s\S]*?</noscript>", "", text, flags=re.IGNORECASE)

    # Phase 2: 删除常见噪声容器
    noisy = r"(nav|footer|header|aside|menu|sidebar|widget|banner|advertisement|ad|promo)"
    text = re.sub(rf"<{noisy}[^>]*>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)

    # Phase 3: 剥离所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # Phase 4: HTML 实体转义
    text = html.unescape(text)

    # Phase 5: 删除 Wikipedia 特有噪声段（截断 References/External links）
    for marker in ("== References ==", "== External links ==", "== See also =="):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # Phase 6: 删除方括号引用、编辑标记
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # Phase 7: 删除 URL
    text = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)

    # Phase 8: 删除 CSS/JS 残留
    text = re.sub(r"\.mw-parser-output[^\n]*", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)

    # Phase 9: 规范化空白
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    return text.strip()


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------

def search_wikipedia(query: str, max_results: int = 1) -> str:
    """搜索 Wikipedia 并返回清洗后的页面内容。

    Args:
        query: 搜索关键词。
        max_results: 返回的最多结果数（默认 1）。

    Returns:
        JSON 字符串：{"title": ..., "content": ...}，或错误说明字符串。
    """
    import json

    try:
        results = _search_pages(query, limit=max_results)
        if not results:
            return json.dumps({"error": f"No Wikipedia results for: {query!r}"})

        title = results[0]["title"]
        raw_html = _fetch_page_html(title)
        if not raw_html:
            return json.dumps({"error": f"Could not fetch page: {title!r}"})

        content = _clean_html(raw_html)
        if len(content) < 50:
            return json.dumps({"error": f"Content too short after cleaning: {title!r}"})

        return json.dumps(
            {"title": title, "content": content[:_MAX_CONTENT_CHARS]},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.warning("search_wikipedia error: %s", e)
        return json.dumps({"error": str(e)})
