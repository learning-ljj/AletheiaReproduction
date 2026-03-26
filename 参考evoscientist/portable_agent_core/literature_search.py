"""Low-coupling literature search service.

This module mirrors the EvoScientist idea of:
1. search first,
2. fetch URLs later,
3. keep tool boundaries simple,
but it removes the DeepAgents / LangChain dependency chain.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from .resilience import RetryPolicy, ValidationIssue, retry_async, run_with_self_correction
from .shared_types import Citation, SearchHit, SearchProvider


@dataclass(slots=True)
class SearchBundle:
    # Final query after normalization or repair.
    query: str
    # Stable list of deduplicated hits.
    hits: list[SearchHit]
    # Provider names help explain fallback behavior.
    providers_used: list[str]


class StaticSearchProvider:
    name = "static"

    def __init__(self, hits: list[SearchHit]) -> None:
        # Keep a fixed list so demos are deterministic.
        self._hits = list(hits)

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        # Return the first N hits exactly as configured.
        return self._hits[:max_results]


class TavilySearchProvider:
    name = "tavily"

    def __init__(self) -> None:
        # Delay importing the SDK until runtime so the module stays lightweight.
        from tavily import TavilyClient

        # The SDK reads TAVILY_API_KEY from the environment, just like the repo.
        self._client = TavilyClient()

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        # Tavily's Python SDK is sync, so move it to a thread.
        raw = await asyncio.to_thread(
            self._client.search,
            query,
            max_results=max_results,
            topic="general",
        )

        # Normalize the SDK response into SearchHit records.
        hits: list[SearchHit] = []
        for item in raw.get("results", []):
            hits.append(
                SearchHit(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "") or item.get("snippet", ""),
                    score=float(item.get("score", 0.0) or 0.0),
                    provider=self.name,
                    metadata=item,
                )
            )
        return hits


def normalize_query(query: str) -> str:
    """Normalize a literature query before it reaches a provider."""

    # Strip surrounding whitespace first.
    cleaned = query.strip()
    # Replace repeated whitespace to reduce accidental provider misses.
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Remove punctuation that often hurts search recall during retries.
    cleaned = cleaned.strip("\"'()[]{}")
    return cleaned


def build_query_repair(
    current_input: dict[str, Any],
    issue: ValidationIssue,
) -> dict[str, Any] | None:
    """Repair a failed search input using simple, deterministic heuristics."""

    # Only repair empty-result style failures.
    if issue.code != "empty_results":
        return None

    # Pull the current query out of the structured input.
    query = str(current_input.get("query", ""))

    # A shorter query is often more stable than a verbose sentence.
    words = [word for word in re.split(r"\s+", query) if word]

    # If the query is already tiny, there is no safe repair to apply.
    if len(words) <= 3:
        return None

    # Keep the first half of the tokens as a conservative fallback query.
    shorter_query = " ".join(words[: max(3, len(words) // 2)])

    # Return a full replacement input object for the next loop round.
    return {
        **current_input,
        "query": shorter_query,
    }


class MultiProviderLiteratureSearch:
    """Search across multiple providers with fallback and self-correction."""

    def __init__(
        self,
        providers: list[SearchProvider],
        retry_policy: RetryPolicy | None = None,
        min_results: int = 3,
    ) -> None:
        # Providers are tried in order, so the caller controls priority.
        self._providers = providers
        # RetryPolicy is shared across provider calls.
        self._retry_policy = retry_policy or RetryPolicy()
        # min_results defines when we can stop the fallback chain early.
        self._min_results = min_results

    async def search(self, query: str, max_results: int = 5) -> SearchBundle:
        """Search providers, deduplicate hits, and self-correct poor queries."""

        # Step logic is nested so the self-correction wrapper can call it repeatedly.
        async def _step(step_input: dict[str, Any]) -> SearchBundle:
            # Normalize on every round because the repair step may change the query.
            normalized_query = normalize_query(str(step_input["query"]))
            # Collect merged hits and provider names across the fallback chain.
            merged_hits: list[SearchHit] = []
            providers_used: list[str] = []

            # Try providers in priority order until enough usable hits exist.
            for provider in self._providers:
                providers_used.append(provider.name)

                async def _provider_call() -> list[SearchHit]:
                    return await provider.search(normalized_query, max_results)

                # Reuse the shared retry helper from the resilience module.
                hits = await retry_async(_provider_call, self._retry_policy)

                # Merge and deduplicate after every provider.
                merged_hits = deduplicate_hits([*merged_hits, *hits])

                # If enough results exist, stop the fallback chain early.
                if len(merged_hits) >= min(max_results, self._min_results):
                    break

            # Rank the merged hits before returning.
            ranked_hits = rank_hits(merged_hits)

            return SearchBundle(
                query=normalized_query,
                hits=ranked_hits[:max_results],
                providers_used=providers_used,
            )

        # Validation rejects empty results so the repair function can simplify the query.
        def _validate(bundle: SearchBundle) -> ValidationIssue | None:
            if not bundle.hits:
                return ValidationIssue(
                    code="empty_results",
                    message="all search providers returned zero hits",
                )
            return None

        # Run the full search through the self-correction loop.
        result = await run_with_self_correction(
            name="literature_search",
            initial_input={"query": query},
            step=_step,
            validate=_validate,
            repair=build_query_repair,
        )

        # Surface failures as ordinary Python exceptions for the caller.
        if not result.success:
            raise RuntimeError(result.content)

        # The successful value is guaranteed to be a SearchBundle.
        return result.value


def deduplicate_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate search hits by URL, then by normalized title."""

    # URL-based keys are preferred because they are usually the most stable.
    seen_urls: set[str] = set()
    # Title fallback helps when providers omit canonical URLs.
    seen_titles: set[str] = set()
    # Preserve order so higher-priority providers stay earlier.
    deduped: list[SearchHit] = []

    for hit in hits:
        # Normalize URL and title for stable matching.
        normalized_url = hit.url.strip().lower()
        normalized_title = re.sub(r"\s+", " ", hit.title.strip().lower())

        # Skip exact URL duplicates first.
        if normalized_url and normalized_url in seen_urls:
            continue

        # If the URL is missing, fall back to title-based deduplication.
        if not normalized_url and normalized_title in seen_titles:
            continue

        # Record the normalized keys we accepted.
        if normalized_url:
            seen_urls.add(normalized_url)
        if normalized_title:
            seen_titles.add(normalized_title)

        # Keep the original hit object for downstream use.
        deduped.append(hit)

    return deduped


def rank_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Apply a lightweight ranking heuristic tuned for literature-style URLs."""

    def _academic_bonus(hit: SearchHit) -> float:
        # Reward URLs that look like papers, archives, or DOI landing pages.
        url = hit.url.lower()
        title = hit.title.lower()
        bonus = 0.0
        if "doi.org" in url or "arxiv.org" in url:
            bonus += 2.0
        if "paper" in title or "benchmark" in title or "survey" in title:
            bonus += 0.5
        return bonus

    # Sort by provider score plus the lightweight academic bonus.
    return sorted(
        hits,
        key=lambda hit: hit.score + _academic_bonus(hit),
        reverse=True,
    )


def citations_from_bundle(bundle: SearchBundle) -> list[Citation]:
    """Convert a SearchBundle into the compact citation objects used by reports."""

    citations: list[Citation] = []
    for hit in bundle.hits:
        citations.append(
            Citation(
                title=hit.title,
                url=hit.url,
                provider=hit.provider,
            )
        )
    return citations
