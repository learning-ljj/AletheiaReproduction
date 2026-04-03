"""Citation reviewer sub-agent for cite path and claim consistency checks."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

from src.memory.problem_memory import ProblemMemory


class CitationReviewerAgent:
    """Review citation paths, claim-source consistency, and metadata quality."""

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
    _LAYER3_HEADER_RE = re.compile(r"(?mi)^##\s*Layer3[^\n]*$")
    _OPENALEX_WORKS_API = "https://api.openalex.org/works"
    _ARXIV_QUERY_API = "http://export.arxiv.org/api/query"
    _ATOM_NS = {
        "atom": "http://www.w3.org/2005/Atom",
    }
    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "have",
        "were", "been", "into", "over", "under", "then", "than", "such",
        "proof", "claim", "lemma", "step", "thus", "therefore", "hence",
    }

    def __init__(self, *, problem_memory: ProblemMemory | None = None):
        self.problem_memory = problem_memory

    def _resolve_cite_path(self, cite_path: str) -> Path:
        # 这段是“路径归一化”：
        # - 绝对路径：直接用；
        # - 以 runs/ 开头：按仓库相对路径解释；
        # - 其它：默认挂到当前 problem run 目录下。
        raw = Path(cite_path)
        if raw.is_absolute():
            return raw

        as_posix = cite_path.replace("\\", "/")
        if as_posix.startswith("runs/"):
            return Path(as_posix)

        if self.problem_memory is not None:
            return (self.problem_memory.run_dir / raw).resolve()

        return raw

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        # 轻量词法切分：只保留字母数字 token，再过滤 stopwords。
        # 目的：给 claim-source 匹配一个便宜、可解释的语义近似。
        tokens = {token.lower() for token in cls._TOKEN_RE.findall(text or "")}
        return {t for t in tokens if t not in cls._STOPWORDS}

    @classmethod
    def _extract_layer3_metadata(cls, content: str) -> dict[str, str]:
        # 只从 "## Layer3" 段提取 key:value，避免误读正文。
        lines = (content or "").splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            if cls._LAYER3_HEADER_RE.match(line.strip()):
                start_idx = idx + 1
                break

        if start_idx is None:
            return {}

        metadata: dict[str, str] = {}
        for line in lines[start_idx:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("## "):
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
        return metadata

    @classmethod
    def _check_claim_support(cls, claim_span: str | None, content: str) -> tuple[bool, str]:
        # 审查策略：
        # 1) 先走“原文精确包含”快路径；
        # 2) 不命中再走 token overlap 近似匹配。
        # 大白话：优先找“原句证据”，找不到才退一步看“词面相似度”。
        claim = (claim_span or "").strip()
        if not claim:
            return True, "Claim span is empty; skipped semantic support check."

        # Fast exact-substring path first.
        if claim in content:
            return True, "Claim span appears verbatim in cited content."

        claim_tokens = cls._tokenize(claim)
        content_tokens = cls._tokenize(content)
        if not claim_tokens or not content_tokens:
            return False, "Insufficient lexical signals for claim-source support check."

        overlap = claim_tokens.intersection(content_tokens)
        ratio = len(overlap) / max(1, len(claim_tokens))
        if ratio >= 0.45:
            return True, f"Token-overlap support passed (ratio={ratio:.2f})."
        return False, f"Token-overlap support failed (ratio={ratio:.2f})."

    @staticmethod
    def _check_metadata_quality(metadata: dict[str, str]) -> tuple[bool, list[str]]:
        # metadata 不是硬失败条件，而是质量评分条件。
        # 即便 metadata 缺失，也可能先给通过但降低 confidence。
        if not metadata:
            return False, ["Layer3 metadata block is missing."]

        warnings: list[str] = []
        if not metadata.get("title"):
            warnings.append("Layer3 metadata missing title.")
        if not metadata.get("authors"):
            warnings.append("Layer3 metadata missing authors.")
        if not (metadata.get("url") or metadata.get("doi") or metadata.get("arxiv_id") or metadata.get("reference")):
            warnings.append("Layer3 metadata missing source locator (url/doi/arxiv_id/reference).")

        return len(warnings) == 0, warnings

    @staticmethod
    def _cascade_identity_hint(metadata: dict[str, str], cite_path: str) -> str:
        doi = (metadata.get("doi") or "").strip()
        arxiv_id = (metadata.get("arxiv_id") or "").strip()
        title = (metadata.get("title") or "").strip()
        if doi:
            return f"Identity cascade anchor: DOI={doi}"
        if arxiv_id:
            return f"Identity cascade anchor: arXiv={arxiv_id}"
        if title:
            return f"Identity cascade anchor: title={title}"
        return f"Identity cascade fallback: cite_path={cite_path}"

    @staticmethod
    def _title_similarity(expected_title: str, observed_title: str) -> float:
        left = (expected_title or "").strip().lower()
        right = (observed_title or "").strip().lower()
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    @staticmethod
    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
        # 统一 HTTP JSON 请求，方便在单点改 UA、超时、追踪策略。
        request_headers = {"User-Agent": "Aletheia-CitationReviewer/1.0"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, headers=request_headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
        import json
        return json.loads(payload)

    @classmethod
    def _verify_doi_via_openalex(
        cls,
        doi: str,
        title: str,
    ) -> tuple[bool | None, str, bool]:
        # DOI 校验是 identity cascade 里的最高优先级。
        # 返回三元组：
        # - status: True/False/None（None 表示跳过或不确定）
        # - detail: 人类可读证据
        # - confident: 这个结论是否足够“硬”
        normalized_doi = (doi or "").strip()
        if not normalized_doi:
            return None, "DOI is empty; skip DOI cascade step.", False

        params = urllib.parse.urlencode({"filter": f"doi:{normalized_doi}", "per-page": 1})
        url = f"{cls._OPENALEX_WORKS_API}?{params}"
        try:
            payload = cls._request_json(url)
        except Exception as exc:  # noqa: BLE001
            return None, f"DOI cascade skipped due network error: {type(exc).__name__}: {exc}", False

        results = payload.get("results") or []
        if not results:
            return False, f"DOI cascade failed: OpenAlex has no record for DOI={normalized_doi}.", True

        observed_title = str((results[0] or {}).get("display_name") or "").strip()
        if title.strip():
            similarity = cls._title_similarity(title, observed_title)
            if similarity < 0.65:
                return (
                    False,
                    f"DOI cascade mismatch: title similarity={similarity:.2f} is below 0.65.",
                    True,
                )
            return True, f"DOI cascade passed via OpenAlex (title similarity={similarity:.2f}).", True

        return True, "DOI cascade passed via OpenAlex record existence.", True

    @classmethod
    def _verify_arxiv_identifier(
        cls,
        arxiv_id: str,
        title: str,
    ) -> tuple[bool | None, str, bool]:
        # arXiv 校验逻辑与 DOI 类似：先看 ID 是否存在，再比标题相似度。
        normalized_arxiv = (arxiv_id or "").strip()
        if not normalized_arxiv:
            return None, "arXiv id is empty; skip arXiv cascade step.", False

        params = urllib.parse.urlencode({"id_list": normalized_arxiv})
        url = f"{cls._ARXIV_QUERY_API}?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "Aletheia-CitationReviewer/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                xml_text = response.read()
        except Exception as exc:  # noqa: BLE001
            return None, f"arXiv cascade skipped due network error: {type(exc).__name__}: {exc}", False

        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", cls._ATOM_NS)
        if not entries:
            return False, f"arXiv cascade failed: no entry for arXiv={normalized_arxiv}.", True

        title_element = entries[0].find("atom:title", cls._ATOM_NS)
        observed_title = (title_element.text or "").strip() if title_element is not None else ""
        if title.strip():
            similarity = cls._title_similarity(title, observed_title)
            if similarity < 0.65:
                return (
                    False,
                    f"arXiv cascade mismatch: title similarity={similarity:.2f} is below 0.65.",
                    True,
                )
            return True, f"arXiv cascade passed (title similarity={similarity:.2f}).", True

        return True, "arXiv cascade passed by identifier existence.", True

    @classmethod
    def _verify_title_openalex(cls, title: str, *, fallback: bool) -> tuple[bool | None, str, bool]:
        # 当 DOI/arXiv 不可用时，用标题近似匹配兜底。
        # fallback 模式阈值更宽松（0.82），主路径更严格（0.88）。
        normalized_title = (title or "").strip()
        if not normalized_title:
            return None, "Title is empty; skip title cascade step.", False

        params = urllib.parse.urlencode({"search": normalized_title, "per-page": 3})
        url = f"{cls._OPENALEX_WORKS_API}?{params}"
        try:
            payload = cls._request_json(url)
        except Exception as exc:  # noqa: BLE001
            return None, f"Title cascade skipped due network error: {type(exc).__name__}: {exc}", False

        candidates = payload.get("results") or []
        if not candidates:
            label = "Title fallback" if fallback else "OpenAlex title step"
            return None, f"{label}: no candidate results.", False

        best_similarity = 0.0
        for candidate in candidates:
            observed = str((candidate or {}).get("display_name") or "").strip()
            best_similarity = max(best_similarity, cls._title_similarity(normalized_title, observed))

        threshold = 0.82 if fallback else 0.88
        if best_similarity >= threshold:
            label = "Title fallback" if fallback else "OpenAlex title step"
            return True, f"{label} passed (best title similarity={best_similarity:.2f}).", True

        label = "Title fallback" if fallback else "OpenAlex title step"
        return None, f"{label} inconclusive (best title similarity={best_similarity:.2f}).", False

    @classmethod
    def _verify_identity_cascade(cls, metadata: dict[str, str]) -> tuple[bool | None, str, bool]:
        # 级联顺序：DOI -> title(openalex) -> arXiv -> title(fallback)
        # 设计目的：优先走“强身份键”，再走“弱身份近似”。
        doi = (metadata.get("doi") or "").strip()
        arxiv_id = (metadata.get("arxiv_id") or "").strip()
        title = (metadata.get("title") or "").strip()

        details: list[str] = []

        if doi:
            status, detail, confident = cls._verify_doi_via_openalex(doi, title)
            details.append(detail)
            if status is not None:
                return status, " | ".join(details), confident

        if title:
            status, detail, confident = cls._verify_title_openalex(title, fallback=False)
            details.append(detail)
            if status is True:
                return True, " | ".join(details), confident

        if arxiv_id:
            status, detail, confident = cls._verify_arxiv_identifier(arxiv_id, title)
            details.append(detail)
            if status is not None:
                return status, " | ".join(details), confident

        if title:
            status, detail, confident = cls._verify_title_openalex(title, fallback=True)
            details.append(detail)
            if status is not None:
                return status, " | ".join(details), confident

        if details:
            return None, " | ".join(details), False
        return None, "Identity cascade skipped: missing DOI/arXiv/title.", False

    def _review_one(self, cite_path: str, claim_span: str | None = None) -> dict:
        # 单条引用审查是一个小流水线：
        # path_exists -> claim_support -> metadata_quality -> identity_cascade。
        resolved = self._resolve_cite_path(cite_path)
        exists = resolved.exists() and resolved.is_file()
        if not exists:
            return {
                "cite": cite_path,
                "resolved_path": str(resolved),
                "passed": False,
                "reason": "PATH_NOT_FOUND",
                "evidence": ["Citation path does not exist on disk."],
                "confidence": 0.0,
                "suggested_action": "fix_cite_path",
            }

        content = resolved.read_text(encoding="utf-8")
        claim_ok, claim_detail = self._check_claim_support(claim_span, content)
        if not claim_ok:
            # claim 对不上来源，直接判失败（这属于实质性问题）。
            return {
                "cite": cite_path,
                "resolved_path": str(resolved),
                "passed": False,
                "reason": "CLAIM_NOT_SUPPORTED",
                "evidence": [claim_detail],
                "confidence": 0.25,
                "suggested_action": "revise_or_replace_claim",
            }

        metadata = self._extract_layer3_metadata(content)
        metadata_ok, metadata_warnings = self._check_metadata_quality(metadata)
        identity_status, identity_detail, identity_confident = self._verify_identity_cascade(metadata)
        if identity_status is False and identity_confident:
            # 身份级联给出“高置信不一致”时，按硬失败处理。
            return {
                "cite": cite_path,
                "resolved_path": str(resolved),
                "passed": False,
                "reason": "IDENTITY_MISMATCH",
                "evidence": [claim_detail, identity_detail],
                "confidence": 0.2,
                "suggested_action": "fix_citation_identity",
                "checks": {
                    "path_exists": True,
                    "claim_source_match": True,
                    "metadata_consistency": metadata_ok,
                    "identity_cascade": False,
                },
            }

        # 通过分支也会带 confidence，便于上层做软门控。
        confidence = 0.95 if metadata_ok else 0.75
        evidence = [claim_detail, self._cascade_identity_hint(metadata, cite_path)]
        if identity_detail:
            evidence.append(identity_detail)
        evidence.extend(metadata_warnings)

        if identity_status is True:
            confidence = min(0.99, confidence + 0.03)

        return {
            "cite": cite_path,
            "resolved_path": str(resolved),
            "passed": True,
            "reason": "OK",
            "evidence": evidence,
            "confidence": confidence,
            "suggested_action": "none" if metadata_ok else "enrich_layer3_metadata",
            "checks": {
                "path_exists": True,
                "claim_source_match": True,
                "metadata_consistency": metadata_ok,
                "identity_cascade": (
                    identity_status
                    if identity_status is not None
                    else "INCONCLUSIVE"
                ),
            },
        }

    def review(self, *, cites: list[str], claim_spans: list[str] | None = None) -> dict:
        # 批处理入口：逐条 review，再汇总 fail_count 和 severity_suggestion。
        if not cites:
            return {
                "summary": "No citations to review.",
                "items": [],
                "fail_count": 0,
                "severity_suggestion": "CORRECT",
            }

        spans = claim_spans or []
        items: list[dict] = []
        fail_count = 0

        for idx, cite in enumerate(cites):
            claim_span = spans[idx] if idx < len(spans) else None
            item = self._review_one(cite, claim_span)
            if not item["passed"]:
                fail_count += 1
            items.append(item)

        passed = len(items) - fail_count
        fail_ratio = fail_count / max(1, len(items))

        # 严重度规则（当前版本）：
        # - 全通过 -> CORRECT
        # - 失败占比 >= 0.5 -> CRITICAL_FLAW
        # - 其它 -> MINOR_FLAW
        if fail_count == 0:
            severity = "CORRECT"
        elif fail_ratio >= 0.5:
            severity = "CRITICAL_FLAW"
        else:
            severity = "MINOR_FLAW"

        summary = (
            f"Reviewed {len(items)} citations: {passed} passed, {fail_count} failed. "
            f"Suggested severity={severity}."
        )
        return {
            "summary": summary,
            "items": items,
            "fail_count": fail_count,
            "severity_suggestion": severity,
        }
