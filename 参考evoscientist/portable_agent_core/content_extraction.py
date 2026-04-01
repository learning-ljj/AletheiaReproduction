"""Low-coupling content extraction service.

The source repository fetches HTML and converts it to Markdown.
This module keeps the same idea but makes the steps explicit:
1. fetch,
2. inspect MIME type,
3. parse HTML or PDF,
4. clean the output.

中文说明：
该模块实现了通用的内容抽取管道，负责从 URL 抓取原始响应，判断 MIME 类型，
并将 HTML / PDF 转换为对模型友好的 Markdown 与纯文本，同时提供简单的
自我修复（例如去掉 URL 参数）逻辑来提升鲁棒性。
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from .resilience import RetryPolicy, ValidationIssue, retry_async, run_with_self_correction
from .shared_types import ExtractedDocument


class HtmlToTextParser(HTMLParser):
    def __init__(self) -> None:
        # Initialize the stdlib parser first.
        super().__init__()
        # Accumulate text chunks in order.
        # parts 列表按顺序累积文本片段，用于生成简单的纯文本输出。
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        # Keep only non-empty text fragments.
        # 仅保留非空文本片段并去除首尾空白。
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        # Join with line breaks so paragraph boundaries stay readable.
        # 以换行拼接，便于保持段落边界的可读性。
        return "\n".join(self.parts)


@dataclass(slots=True)
class FetchedResponse:
    # Raw bytes are needed for PDF parsing.
    # Raw bytes：用于 PDF 解析器（按页提取文本）。
    content: bytes
    # Decoded text is needed for HTML parsing.
    # 解码后的文本：用于 HTML 路径下的解析。
    text: str
    # Headers are normalized into a plain dictionary.
    # 响应头字典，键通常小写化以便查找 content-type 等字段。
    headers: dict[str, str]


class UniversalContentExtractor:
    """Extract model-friendly content from HTML or PDF URLs."""

    # UniversalContentExtractor 提供 fetch + parse + validate 的完整流程，
    # 并使用 resilience 模块的自我修正回路来处理常见失败。

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        # HTTP timeout should stay explicit because extraction failures are common.
        # HTTP 超时（秒），在网络或站点响应慢时避免长时间挂起。
        self._timeout_seconds = timeout_seconds
        # Reuse the same bounded retry policy style as the other modules.
        # 重试策略复用 resilience 模块提供的 RetryPolicy。
        self._retry_policy = retry_policy or RetryPolicy()

    async def extract(self, source: str) -> ExtractedDocument:
        """Fetch a remote document, parse it, and return a normalized payload."""

        # 将抓取 + 解析步骤封装为一个可重入的 step，以便 run_with_self_correction 在
        # 校验失败时能传入修复后的输入并重试。
        async def _step(step_input: dict[str, str]) -> ExtractedDocument:
            # Fetch the raw response bytes first.
            response = await self._fetch(step_input["source"])
            # Derive the MIME type from the response headers.
            mime_type = response.headers.get("content-type", "").split(";")[0].strip()

            # Route HTML and PDF through different parsers.
            if mime_type == "application/pdf" or source.lower().endswith(".pdf"):
                return self._extract_pdf(source, response.content, mime_type)

            # Default to HTML because many servers return text/html or omit MIME.
            return self._extract_html(source, response.text, mime_type or "text/html")

        # Validation rejects empty extraction so the repair step can retry with a URL cleanup.
        def _validate(document: ExtractedDocument) -> ValidationIssue | None:
            if not document.text.strip() and not document.markdown.strip():
                return ValidationIssue(
                    code="empty_document",
                    message="document parser returned no readable content",
                )
            return None

        # Repair removes tracking query parameters from the URL when extraction fails.
        def _repair(step_input: dict[str, str], issue: ValidationIssue) -> dict[str, str] | None:
            if issue.code != "empty_document":
                return None
            source_url = step_input["source"]
            if "?" not in source_url:
                return None
            return {
                "source": source_url.split("?", 1)[0],
            }

        # Run fetch + parse + validate through the self-correction loop.
        result = await run_with_self_correction(
            name="content_extraction",
            initial_input={"source": source},
            step=_step,
            validate=_validate,
            repair=_repair,
        )

        # Bubble failures up as Python exceptions to keep the caller simple.
        if not result.success:
            raise RuntimeError(result.content)

        return result.value

    async def _fetch(self, source: str) -> FetchedResponse:
        """Fetch a remote document with retries."""

        # Use a browser-like user agent because many sites reject default clients.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }

        async def _operation() -> FetchedResponse:
            try:
                # 优先使用 httpx，因为其 async API 更简洁且支持超时与重定向控制。
                import httpx

                # Open a short-lived client so the extractor remains self-contained.
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    # 使用配置的超时执行 GET 请求。
                    response = await client.get(
                        source,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )
                    # Raise on 4xx and 5xx so retries can happen.
                    response.raise_for_status()
                    return FetchedResponse(
                        content=response.content,
                        text=response.text,
                        headers=dict(response.headers),
                    )
            except ModuleNotFoundError:
                # Fall back to urllib so the demo still works in minimal environments.
                from urllib.request import Request, urlopen

                # 备用实现：在缺少 httpx 时使用标准库的 urllib，同样支持超时。
                def _blocking_fetch() -> FetchedResponse:
                    request = Request(source, headers=headers)
                    with urlopen(request, timeout=self._timeout_seconds) as response:
                        content = response.read()
                        encoding = response.headers.get_content_charset() or "utf-8"
                        text = content.decode(encoding, errors="replace")
                        return FetchedResponse(
                            content=content,
                            text=text,
                            headers={k.lower(): v for k, v in response.headers.items()},
                        )

                return await asyncio.to_thread(_blocking_fetch)

        # Reuse the generic retry helper from the resilience module.
        return await retry_async(_operation, self._retry_policy)

    def _extract_html(
        self,
        source: str,
        html: str,
        mime_type: str,
    ) -> ExtractedDocument:
        """Convert HTML into Markdown and plain text."""

        # Try markdownify first because it matches the source repository.
        try:
            from markdownify import markdownify

            markdown = markdownify(html)
        except Exception:
            # Fall back to a tiny stdlib HTML parser when markdownify is absent.
            parser = HtmlToTextParser()
            parser.feed(html)
            markdown = parser.text()

        # Convert Markdown into cleaner plain text for validation or chunking.
        text = clean_text(markdown)
        # Try to recover a title from the HTML head.
        title = extract_title_from_html(html)

        return ExtractedDocument(
            source=source,
            mime_type=mime_type,
            title=title,
            markdown=clean_markdown(markdown),
            text=text,
            metadata={"parser": "html"},
        )

    def _extract_pdf(
        self,
        source: str,
        content: bytes,
        mime_type: str,
    ) -> ExtractedDocument:
        """Extract text from a PDF if pypdf is installed."""

        try:
            from pypdf import PdfReader
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PDF extraction requires the optional 'pypdf' package"
            ) from exc

        # 将原始字节包装为 file-like 对象以供 PdfReader 使用。
        reader = PdfReader(io.BytesIO(content))
        # Pull text page by page to preserve natural order.
        page_texts = [page.extract_text() or "" for page in reader.pages]
        # Join pages with double breaks so sections stay readable.
        text = clean_text("\n\n".join(page_texts))
        # Derive a best-effort title from PDF metadata.
        title = ""
        if reader.metadata:
            title = str(reader.metadata.get("/Title", "") or "")

        return ExtractedDocument(
            source=source,
            mime_type=mime_type or "application/pdf",
            title=title,
            markdown=text,
            text=text,
            metadata={"parser": "pdf", "pages": len(reader.pages)},
        )


def extract_title_from_html(html: str) -> str:
    """Best-effort HTML title extraction without extra dependencies."""

    # Use a simple regex because the title tag is small and predictable.
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    # Normalize whitespace because titles often contain line breaks.
    return re.sub(r"\s+", " ", match.group(1)).strip()


# clean_markdown / clean_text 提供了一些保守的文本清理规则，目的是在把文本
# 送入 LLM 前去除噪声（重复空行、导航条残留、表格间距等），而不是做深度格式化。


def clean_markdown(markdown: str) -> str:
    """Normalize noisy Markdown before it reaches an LLM."""

    # Collapse excessive blank lines first.
    cleaned = re.sub(r"\n{3,}", "\n\n", markdown)
    # Remove repeated spaces that usually come from nav bars or tables.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def clean_text(text: str) -> str:
    """Convert Markdown-like text into a compact plain-text body."""

    # Strip markdown headers and list markers conservatively.
    cleaned = re.sub(r"^[#>\-\*\s]+", "", text, flags=re.MULTILINE)
    # Collapse many blank lines into one readable paragraph break.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Collapse repeated spaces inside lines.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
