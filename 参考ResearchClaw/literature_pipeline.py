"""Low-coupling literature retrieval, PDF extraction, and summarization reference.

This file deliberately avoids importing the original repo internals.
You can copy it into another agent framework and only wire three surfaces:
1. StableLiteratureService.search(...)
2. PdfExtractor.extract(...)
3. ExtractivePaperSummarizer.summarize(...)
"""

from __future__ import annotations

# `json` is used to decode Semantic Scholar responses.
import json
# `re` is used for query normalization and light section parsing.
import re
# `time` is used for retry backoff when an upstream API rate-limits.
import time
# `uuid` is used to generate stable cache file names without collisions.
import uuid
# `dataclass` keeps the state plain and serializable.
from dataclasses import asdict, dataclass, field
# `Path` makes filesystem handling portable and explicit.
from pathlib import Path
# `Any` and `Optional` make the public surface easier to read.
from typing import Any, Iterable, Optional
# `parse`, `request`, and `error` let us stay on the Python standard library for HTTP.
from urllib import error, parse, request
# `ET` is enough to parse the ArXiv Atom feed.
import xml.etree.ElementTree as ET

# ArXiv exposes Atom XML, so we keep the namespace in one place.
ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class RetryPolicy:
    # Maximum number of attempts, including the first request.
    max_attempts: int = 4
    # Base sleep in seconds for exponential backoff.
    backoff_base: float = 1.5
    # Upper bound to stop retry waits from growing forever.
    max_sleep_seconds: float = 20.0
    # Request timeout passed to the HTTP client.
    timeout_seconds: float = 20.0


@dataclass
class PaperRecord:
    # Human readable paper title.
    title: str
    # Canonical paper identifier from the upstream source.
    paper_id: str
    # Name of the source system, for example `arxiv` or `semantic_scholar`.
    source: str
    # Original user query that produced this paper.
    query: str
    # Paper abstract or summary snippet from the search backend.
    abstract: str = ""
    # Ordered author list.
    authors: list[str] = field(default_factory=list)
    # Publication year if the backend exposes it.
    year: Optional[int] = None
    # DOI if one exists.
    doi: str = ""
    # ArXiv id if one exists.
    arxiv_id: str = ""
    # Landing page URL.
    paper_url: str = ""
    # Direct PDF URL if one exists.
    pdf_url: str = ""
    # Raw backend payload for debugging or later extension.
    raw: dict[str, Any] = field(default_factory=dict)

    def stable_key(self) -> str:
        # DOI is the best cross-source dedup key, so prefer it when present.
        if self.doi:
            return f"doi:{self.doi.lower()}"
        # ArXiv id is stable across ArXiv and Semantic Scholar mirrors.
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        # Otherwise fall back to a normalized title fingerprint.
        normalized_title = normalize_space(self.title).lower()
        return f"title:{normalized_title}"

    def to_dict(self) -> dict[str, Any]:
        # `asdict` keeps the object easy to serialize into JSON or store layers.
        return asdict(self)


@dataclass
class SearchResult:
    # Backend name that produced this result set.
    source: str
    # Original query.
    query: str
    # Returned papers after backend-side parsing.
    papers: list[PaperRecord]
    # Total item count after parsing.
    total: int
    # Soft failures collected while combining multiple sources.
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractedPaper:
    # Final resolved PDF path on disk.
    source_path: str
    # Best-effort title, often taken from metadata or the first heading.
    title: str
    # Full extracted text used by the summarizer.
    text: str
    # Number of pages actually parsed.
    page_count: int
    # Lightweight section map derived from the extracted text.
    sections: dict[str, str] = field(default_factory=dict)
    # Best-effort references list parsed from the end of the paper.
    references: list[str] = field(default_factory=list)
    # Raw PDF metadata for debugging.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # This keeps the object portable across notebooks, services, and tests.
        return asdict(self)


@dataclass
class PaperSummary:
    # Title copied from the extracted paper for convenience.
    title: str
    # One short plain-language summary.
    short_summary: str
    # Lead paragraphs or abstract-like text.
    abstract_like: str
    # Key points collected from the most informative sections.
    key_points: list[str]
    # Method-focused snippet if detected.
    method_summary: str
    # Result-focused snippet if detected.
    result_summary: str
    # Candidate claims you can feed into a workflow/claim graph.
    candidate_claims: list[str]

    def to_dict(self) -> dict[str, Any]:
        # JSON-friendly export for your own store layer.
        return asdict(self)


class SearchBackend:
    # Backend classes override this with a readable source name.
    name = "base"

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        # Subclasses must implement the retrieval call and parsing.
        raise NotImplementedError


def normalize_space(text: str) -> str:
    # Normalize repeated whitespace so downstream comparisons stay stable.
    return re.sub(r"\s+", " ", text or "").strip()


def build_headers(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    # A clear User-Agent helps avoid anonymous-client throttling on some providers.
    headers = {"User-Agent": "migration-reference/1.0 (+https://example.local)"}
    # Optional headers such as an API key are layered on top.
    if extra:
        headers.update(extra)
    return headers


def parse_retry_after(value: Optional[str]) -> float:
    # Missing headers simply mean we will use exponential backoff instead.
    if not value:
        return 0.0
    try:
        # Some providers return a plain integer number of seconds.
        return float(value)
    except (TypeError, ValueError):
        # We intentionally keep this parser minimal for easy portability.
        return 0.0


def should_retry(status_code: int) -> bool:
    # 429 is a direct rate-limit signal.
    if status_code == 429:
        return True
    # Retry temporary upstream failures, but not client-side request errors.
    return 500 <= status_code < 600


def http_get_bytes(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retry_policy: Optional[RetryPolicy] = None,
) -> bytes:
    # Use the default retry configuration unless the caller overrides it.
    policy = retry_policy or RetryPolicy()
    # Encode query parameters once so the retry loop only replays the same request.
    encoded_url = f"{url}?{parse.urlencode(params or {}, doseq=True)}" if params else url
    # Keep the last exception so we can raise a useful error after retries are exhausted.
    last_error: Optional[Exception] = None

    for attempt in range(policy.max_attempts):
        # Build a fresh request object for each attempt.
        req = request.Request(encoded_url, headers=build_headers(headers), method="GET")
        try:
            # `urlopen` is enough here because we only need GET requests.
            with request.urlopen(req, timeout=policy.timeout_seconds) as response:
                # Return raw bytes so callers can decide how to decode them.
                return response.read()
        except error.HTTPError as exc:
            # Retry only for explicit transient status codes.
            if should_retry(exc.code) and attempt < policy.max_attempts - 1:
                # Honor `Retry-After` when the server provides it.
                retry_after = parse_retry_after(exc.headers.get("Retry-After"))
                # Fall back to exponential backoff when the header is absent.
                sleep_seconds = retry_after or (policy.backoff_base * (2 ** attempt))
                # Clamp the sleep so one bad upstream does not stall the whole app.
                time.sleep(min(max(1.0, sleep_seconds), policy.max_sleep_seconds))
                continue
            # Preserve the HTTP error for the final raise path.
            last_error = exc
            break
        except error.URLError as exc:
            # Network wobble is also worth retrying.
            if attempt < policy.max_attempts - 1:
                sleep_seconds = policy.backoff_base * (2 ** attempt)
                time.sleep(min(max(1.0, sleep_seconds), policy.max_sleep_seconds))
                continue
            # Preserve the final connection error if all attempts fail.
            last_error = exc
            break

    # Convert exhausted retries into one explicit runtime error for the caller.
    raise RuntimeError(f"HTTP GET failed for {encoded_url}: {last_error}")


def http_get_text(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retry_policy: Optional[RetryPolicy] = None,
    encoding: str = "utf-8",
) -> str:
    # Decode bytes into text using the upstream API's documented encoding.
    return http_get_bytes(
        url,
        params=params,
        headers=headers,
        retry_policy=retry_policy,
    ).decode(encoding, errors="replace")


def http_get_json(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retry_policy: Optional[RetryPolicy] = None,
) -> dict[str, Any]:
    # JSON APIs become ordinary Python dictionaries here.
    payload = http_get_text(url, params=params, headers=headers, retry_policy=retry_policy)
    return json.loads(payload)


class SemanticScholarSearchBackend(SearchBackend):
    # Match the upstream provider name used in the original repo.
    name = "semantic_scholar"

    def __init__(self, api_key: str = "", retry_policy: Optional[RetryPolicy] = None) -> None:
        # The API key is optional but recommended for better rate limits.
        self.api_key = api_key.strip()
        # Reuse one retry policy object so behaviour stays consistent.
        self.retry_policy = retry_policy or RetryPolicy()

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        # Semantic Scholar lets us request only the fields we need.
        fields = [
            "title",
            "abstract",
            "year",
            "authors",
            "url",
            "externalIds",
            "openAccessPdf",
        ]
        # The graph search endpoint is the repo's main stable metadata source.
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # Keep query parameters flat for easy logging and testing.
        params = {"query": query, "limit": max_results, "fields": ",".join(fields)}
        # Add the API key only when the caller configured one.
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        # Fetch and parse the JSON payload.
        payload = http_get_json(url, params=params, headers=headers, retry_policy=self.retry_policy)

        papers: list[PaperRecord] = []
        for item in payload.get("data", []):
            # Author records are nested dictionaries, so flatten them early.
            authors = [normalize_space(author.get("name", "")) for author in item.get("authors", [])]
            # Semantic Scholar exposes cross-source identifiers under `externalIds`.
            external_ids = item.get("externalIds") or {}
            # Prefer the upstream open-access PDF; if absent, synthesize ArXiv PDF URLs.
            open_access_pdf = (item.get("openAccessPdf") or {}).get("url", "")
            arxiv_id = normalize_space(external_ids.get("ArXiv", ""))
            pdf_url = open_access_pdf or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "")
            # Build a normalized record for higher layers.
            papers.append(
                PaperRecord(
                    title=normalize_space(item.get("title", "Untitled")),
                    paper_id=normalize_space(item.get("paperId", "")) or str(uuid.uuid4()),
                    source=self.name,
                    query=query,
                    abstract=normalize_space(item.get("abstract", "")),
                    authors=[author for author in authors if author],
                    year=item.get("year"),
                    doi=normalize_space(external_ids.get("DOI", "")),
                    arxiv_id=arxiv_id,
                    paper_url=normalize_space(item.get("url", "")),
                    pdf_url=pdf_url,
                    raw=item,
                )
            )

        # Wrap the result so callers can inspect the source and any soft errors.
        return SearchResult(source=self.name, query=query, papers=papers, total=len(papers))


class ArxivSearchBackend(SearchBackend):
    # Match the original repo terminology.
    name = "arxiv"

    def __init__(self, retry_policy: Optional[RetryPolicy] = None) -> None:
        # ArXiv is sensitive to aggressive scraping, so retry policy matters.
        self.retry_policy = retry_policy or RetryPolicy()

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        # ArXiv uses an Atom feed instead of JSON.
        url = "http://export.arxiv.org/api/query"
        # `all:` is the simplest general-purpose field for migration use.
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        # Fetch the Atom XML as text.
        xml_text = http_get_text(url, params=params, retry_policy=self.retry_policy)
        # Parse the XML document into a tree.
        root = ET.fromstring(xml_text)

        papers: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
            # Extract the canonical entry URL, for example `http://arxiv.org/abs/2501.01234v1`.
            entry_id = normalize_space(entry.findtext("atom:id", default="", namespaces=ARXIV_ATOM_NS))
            # Keep only the final path segment as the upstream paper identifier.
            paper_id = entry_id.rsplit("/", 1)[-1]
            # Strip the version suffix so the id is stable across updates.
            arxiv_id = re.sub(r"v\d+$", "", paper_id)
            # Titles and abstracts often contain newlines in Atom feeds, so normalize them.
            title = normalize_space(entry.findtext("atom:title", default="", namespaces=ARXIV_ATOM_NS))
            abstract = normalize_space(entry.findtext("atom:summary", default="", namespaces=ARXIV_ATOM_NS))
            # Publication years are encoded in the `published` timestamp.
            published = normalize_space(entry.findtext("atom:published", default="", namespaces=ARXIV_ATOM_NS))
            year = int(published[:4]) if published[:4].isdigit() else None
            # Collect author names in display order.
            authors = [
                normalize_space(author.findtext("atom:name", default="", namespaces=ARXIV_ATOM_NS))
                for author in entry.findall("atom:author", ARXIV_ATOM_NS)
            ]
            # The PDF URL is stable for ArXiv records and easy to synthesize.
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
            # Build a shared paper model so downstream code is backend-agnostic.
            papers.append(
                PaperRecord(
                    title=title or "Untitled",
                    paper_id=paper_id or str(uuid.uuid4()),
                    source=self.name,
                    query=query,
                    abstract=abstract,
                    authors=[author for author in authors if author],
                    year=year,
                    arxiv_id=arxiv_id,
                    paper_url=entry_id,
                    pdf_url=pdf_url,
                    raw={"published": published},
                )
            )

        # The return shape intentionally matches Semantic Scholar.
        return SearchResult(source=self.name, query=query, papers=papers, total=len(papers))


class StableLiteratureService:
    """Combine multiple paper sources with retry, fallback, and deduplication."""

    def __init__(self, backends: Iterable[SearchBackend]) -> None:
        # Materialize the iterable once because the service may call it more than once.
        self.backends = list(backends)

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        sources: Optional[set[str]] = None,
    ) -> SearchResult:
        # Keep all parsed papers here before cross-source deduplication.
        aggregated: list[PaperRecord] = []
        # Collect soft failures so one source does not hide the others.
        errors: list[str] = []
        # Restrict the active backends if the caller asks for specific sources.
        active_backends = [backend for backend in self.backends if not sources or backend.name in sources]

        for backend in active_backends:
            try:
                # Ask each backend for up to `max_results`; dedup happens later.
                result = backend.search(query=query, max_results=max_results)
                aggregated.extend(result.papers)
            except Exception as exc:
                # Record the error and move on to the next backend.
                errors.append(f"{backend.name}: {exc}")

        # Deduplicate after aggregation so different sources can complement each other.
        deduped = deduplicate_papers(aggregated)
        # Prefer papers that expose more metadata when the dedup key collides.
        deduped = rank_papers(deduped)
        # Apply the final global cap only once.
        deduped = deduped[:max_results]
        # Expose the combined result under a synthetic source name.
        return SearchResult(source="combined", query=query, papers=deduped, total=len(deduped), errors=errors)


def paper_quality_score(paper: PaperRecord) -> int:
    # More metadata usually means a more reusable search result.
    score = 0
    # Abstracts are essential for ranking and later summarization.
    if paper.abstract:
        score += 3
    # A PDF URL saves a later retrieval hop.
    if paper.pdf_url:
        score += 2
    # DOI is the strongest identity field.
    if paper.doi:
        score += 2
    # Author and year make the result more trustworthy for human review.
    if paper.authors:
        score += 1
    if paper.year:
        score += 1
    return score


def deduplicate_papers(papers: Iterable[PaperRecord]) -> list[PaperRecord]:
    # The map keeps only the best paper per stable identity key.
    best_by_key: dict[str, PaperRecord] = {}
    for paper in papers:
        # Compute the cross-source identity key.
        key = paper.stable_key()
        # Keep the richer record when duplicates collide.
        existing = best_by_key.get(key)
        if existing is None or paper_quality_score(paper) > paper_quality_score(existing):
            best_by_key[key] = paper
    # Preserve insertion order from Python dictionaries for deterministic results.
    return list(best_by_key.values())


def rank_papers(papers: Iterable[PaperRecord]) -> list[PaperRecord]:
    # Rank by metadata quality first, then by year, then by title for deterministic output.
    return sorted(
        papers,
        key=lambda paper: (
            paper_quality_score(paper),
            paper.year or 0,
            normalize_space(paper.title).lower(),
        ),
        reverse=True,
    )

class PdfExtractor:
    """Resolve a paper source into a local PDF and extract plain text."""

    def __init__(self, cache_dir: str | Path = "./.paper_cache", retry_policy: Optional[RetryPolicy] = None) -> None:
        # The cache prevents repeated downloads when a workflow is resumed.
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        # Ensure the cache exists before the first extraction call.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Reuse the same retry settings as the search backends by default.
        self.retry_policy = retry_policy or RetryPolicy()

    def resolve_source(self, source: str) -> Path:
        # Local files are the cheapest path, so try them first.
        local_path = Path(source).expanduser()
        if local_path.exists():
            return local_path.resolve()

        # Recognize raw ArXiv ids such as `2501.01234` or `2501.01234v2`.
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source.strip()):
            # Strip the version so the cache key is stable.
            arxiv_id = re.sub(r"v\d+$", "", source.strip())
            return self.download_file(f"https://arxiv.org/pdf/{arxiv_id}.pdf", filename=f"{arxiv_id}.pdf")

        # Generic HTTP(S) sources are treated as downloadable PDFs.
        if source.startswith("http://") or source.startswith("https://"):
            return self.download_file(source)

        # Anything else is ambiguous and should fail loudly for the caller.
        raise FileNotFoundError(f"Cannot resolve paper source: {source}")

    def download_file(self, url: str, filename: str = "") -> Path:
        # Reuse a friendly name when possible so the cache stays human-readable.
        chosen_name = filename or self._filename_from_url(url)
        # Store everything under the extractor cache directory.
        destination = self.cache_dir / chosen_name
        # Skip the network if we already downloaded the file before.
        if destination.exists():
            return destination
        # Fetch the binary PDF payload with the same retry logic used for APIs.
        payload = http_get_bytes(url, retry_policy=self.retry_policy)
        # Write atomically enough for a single-process migration reference.
        destination.write_bytes(payload)
        return destination

    def _filename_from_url(self, url: str) -> str:
        # Read the final path segment from the URL, for example `2501.01234.pdf`.
        raw_name = Path(parse.urlparse(url).path).name
        # Fall back to a random file name if the URL path is empty.
        if not raw_name:
            raw_name = f"paper-{uuid.uuid4().hex}.pdf"
        # Ensure the extension is present so PDF tools can infer the type.
        if not raw_name.lower().endswith(".pdf"):
            raw_name = f"{raw_name}.pdf"
        return raw_name

    def extract(self, source: str, *, max_pages: Optional[int] = None) -> ExtractedPaper:
        # Resolve local path, URL, or ArXiv id into one concrete file path.
        pdf_path = self.resolve_source(source)

        try:
            # `pdfplumber` extracts layout-aware text and is usually the better first choice.
            import pdfplumber  # type: ignore  # noqa: F401

            return self._extract_with_pdfplumber(pdf_path, max_pages=max_pages)
        except ImportError:
            # Fall back cleanly when the optional dependency is absent.
            pass

        try:
            # `PyPDF2` is a lighter fallback available in many environments.
            import PyPDF2  # type: ignore  # noqa: F401

            return self._extract_with_pypdf2(pdf_path, max_pages=max_pages)
        except ImportError:
            # Make the missing dependency explicit so the caller knows how to proceed.
            raise RuntimeError("Install `pdfplumber` or `PyPDF2` to enable PDF text extraction.")

    def _extract_with_pdfplumber(self, pdf_path: Path, *, max_pages: Optional[int] = None) -> ExtractedPaper:
        # Import inside the method so the module remains optional.
        import pdfplumber  # type: ignore

        # Accumulate page text before post-processing.
        page_texts: list[str] = []
        # Capture metadata from the PDF container when available.
        metadata: dict[str, Any] = {}

        with pdfplumber.open(str(pdf_path)) as pdf:
            # `metadata` can be `None`, so normalize it into a dictionary.
            metadata = dict(pdf.metadata or {})
            # Respect `max_pages` so callers can trade off latency versus completeness.
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                # `extract_text` may return `None` on image-only pages.
                text = page.extract_text() or ""
                # Keep only non-empty pages to reduce downstream noise.
                if normalize_space(text):
                    page_texts.append(text)

        # Join page texts into one document-level string.
        full_text = "\n\n".join(page_texts)
        # Try metadata title first, then fall back to the document heading.
        title = normalize_space(str(metadata.get("Title", ""))) or guess_title_from_text(full_text) or pdf_path.stem
        # Build lightweight section and reference helpers for summarization.
        sections = extract_named_sections(full_text)
        references = extract_reference_list(full_text)
        return ExtractedPaper(
            source_path=str(pdf_path),
            title=title,
            text=full_text,
            page_count=len(page_texts),
            sections=sections,
            references=references,
            metadata=metadata,
        )

    def _extract_with_pypdf2(self, pdf_path: Path, *, max_pages: Optional[int] = None) -> ExtractedPaper:
        # Import inside the method so the module remains optional.
        import PyPDF2  # type: ignore

        # This list collects text from each parsed page.
        page_texts: list[str] = []
        # Metadata is usually a document information dictionary.
        metadata: dict[str, Any] = {}

        with pdf_path.open("rb") as file_obj:
            # Create the reader from the binary PDF stream.
            reader = PyPDF2.PdfReader(file_obj)
            # Normalize metadata into a plain dictionary for serialization.
            metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items()}
            # Apply the same page limit behaviour as the `pdfplumber` path.
            pages = reader.pages[:max_pages] if max_pages else reader.pages
            for page in pages:
                # `extract_text` may also return `None` here.
                text = page.extract_text() or ""
                # Keep only useful page text.
                if normalize_space(text):
                    page_texts.append(text)

        # Assemble the whole paper text after page extraction.
        full_text = "\n\n".join(page_texts)
        # Reuse the same title resolution logic for consistency.
        title = normalize_space(metadata.get("/Title", "")) or guess_title_from_text(full_text) or pdf_path.stem
        # Reuse the same section and reference heuristics.
        sections = extract_named_sections(full_text)
        references = extract_reference_list(full_text)
        return ExtractedPaper(
            source_path=str(pdf_path),
            title=title,
            text=full_text,
            page_count=len(page_texts),
            sections=sections,
            references=references,
            metadata=metadata,
        )


def guess_title_from_text(text: str) -> str:
    # Look at the first few non-empty lines because titles usually appear at the top.
    for line in text.splitlines()[:12]:
        # Normalize line whitespace before inspection.
        candidate = normalize_space(line)
        # Skip very short lines because they are often page numbers or headers.
        if len(candidate) < 15:
            continue
        # Skip obvious author lines and abstract markers.
        if candidate.lower() in {"abstract", "introduction"}:
            continue
        return candidate
    # Return an empty title when no likely heading is found.
    return ""


def extract_named_sections(text: str) -> dict[str, str]:
    # Common academic section names are enough for a portable baseline.
    section_names = [
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "approach",
        "experiment",
        "experiments",
        "evaluation",
        "results",
        "discussion",
        "conclusion",
        "limitations",
        "references",
    ]
    # Split into lines once so we can scan forward efficiently.
    lines = text.splitlines()
    # Record candidate section start lines.
    starts: list[tuple[str, int]] = []

    for index, raw_line in enumerate(lines):
        # Normalize each line before section matching.
        line = normalize_space(raw_line).lower().strip(":")
        # Ignore empty lines because they can never be section headers.
        if not line:
            continue
        # Match either an exact section name or a numbered section heading.
        for section_name in section_names:
            if line == section_name or re.fullmatch(rf"\d+(\.\d+)*\s+{re.escape(section_name)}", line):
                starts.append((section_name, index))
                break

    # Nothing matched, so return an empty map instead of guessing wildly.
    if not starts:
        return {}

    sections: dict[str, str] = {}
    for idx, (section_name, start_line) in enumerate(starts):
        # The section ends where the next section begins.
        end_line = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        # Slice the original lines so we keep the original formatting as much as possible.
        body = "\n".join(lines[start_line:end_line]).strip()
        # Store only non-empty section bodies.
        if body:
            sections[section_name] = body
    return sections


def extract_reference_list(text: str) -> list[str]:
    # Find the `References` heading near the end of the paper.
    match = re.search(r"(?is)\n\s*(references|bibliography)\s*\n", text)
    # Without a heading, the heuristic is too unreliable to continue.
    if not match:
        return []
    # Keep only the tail section after the references heading.
    tail = text[match.end() :]
    # Split on blank lines because many PDFs flatten citations into paragraphs.
    chunks = [normalize_space(chunk) for chunk in re.split(r"\n\s*\n", tail)]
    # Keep medium-length chunks that look like citations.
    references = [chunk for chunk in chunks if 20 <= len(chunk) <= 800]
    # Cap the list because huge references sections add little value during migration.
    return references[:30]

class ExtractivePaperSummarizer:
    """Generate a lightweight summary without depending on an LLM."""

    def summarize(self, paper: ExtractedPaper, *, max_points: int = 5) -> PaperSummary:
        # Grab the beginning of the paper because titles and abstracts usually live there.
        abstract_like = best_section(paper, ["abstract", "introduction"], fallback_chars=1600)
        # Prefer explicit method sections when present.
        method_summary = best_section(paper, ["method", "methods", "approach"], fallback_chars=1200)
        # Prefer explicit result sections when present.
        result_summary = best_section(paper, ["results", "evaluation", "discussion", "conclusion"], fallback_chars=1200)
        # Build short bullet-like points from the most informative snippets.
        key_points = build_key_points([abstract_like, method_summary, result_summary], limit=max_points)
        # Produce claim candidates that can be turned into claim objects later.
        candidate_claims = build_candidate_claims(key_points)
        # Fuse the main snippets into a single short paragraph for UI display.
        short_summary = truncate_text(" ".join([abstract_like, result_summary]).strip(), max_chars=500)
        return PaperSummary(
            title=paper.title,
            short_summary=short_summary,
            abstract_like=abstract_like,
            key_points=key_points,
            method_summary=method_summary,
            result_summary=result_summary,
            candidate_claims=candidate_claims,
        )


def best_section(paper: ExtractedPaper, names: list[str], *, fallback_chars: int) -> str:
    # Use explicit parsed sections first because they are cleaner than raw page text.
    for name in names:
        body = normalize_space(paper.sections.get(name, ""))
        if body:
            return truncate_text(body, max_chars=fallback_chars)
    # Fall back to the front of the document when the section parser finds nothing.
    return truncate_text(normalize_space(paper.text), max_chars=fallback_chars)


def build_key_points(snippets: list[str], *, limit: int) -> list[str]:
    # Keep key points unique while preserving order.
    points: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        # Split on sentence boundaries for a simple extractive summary.
        sentences = re.split(r"(?<=[.!?])\s+", normalize_space(snippet))
        for sentence in sentences:
            # Clean up the candidate sentence before screening it.
            candidate = truncate_text(normalize_space(sentence), max_chars=220)
            # Skip trivial or header-like fragments.
            if len(candidate) < 40:
                continue
            # Use a lowercase fingerprint to remove near-identical duplicates.
            fingerprint = candidate.lower()
            if fingerprint in seen:
                continue
            points.append(candidate)
            seen.add(fingerprint)
            if len(points) >= limit:
                return points
    return points


def build_candidate_claims(key_points: list[str]) -> list[str]:
    # Turn summary points into claim-like statements that a workflow can track.
    claims: list[str] = []
    for point in key_points:
        # Remove leading discourse markers so the claim reads more directly.
        cleaned = re.sub(r"^(in this paper|we show that|this paper shows that)\s+", "", point, flags=re.I)
        # Keep only sentences that are assertive enough to act as claims.
        if any(marker in cleaned.lower() for marker in ["improve", "outperform", "reduce", "increase", "show", "demonstrate"]):
            claims.append(cleaned)
    # If nothing matched, fall back to the first key point so the caller still has one claim candidate.
    return claims or key_points[:1]


def truncate_text(text: str, *, max_chars: int) -> str:
    # Normalize whitespace before truncation so limits behave predictably.
    normalized = normalize_space(text)
    # Short enough strings are returned as-is.
    if len(normalized) <= max_chars:
        return normalized
    # Cut cleanly and keep an ellipsis to signal truncation.
    return normalized[: max_chars - 1].rstrip() + "..."


if __name__ == "__main__":
    # This block doubles as a smoke test and a copy-paste usage example.
    service = StableLiteratureService(
        backends=[
            SemanticScholarSearchBackend(api_key=""),
            ArxivSearchBackend(),
        ]
    )
    # Replace the query with your own topic when running this file directly.
    result = service.search("retrieval augmented generation", max_results=5)
    # Print one compact line per paper so the output stays readable.
    for paper in result.papers:
        print(json.dumps(paper.to_dict(), ensure_ascii=False))
