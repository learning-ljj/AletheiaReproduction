"""Shared data models for the migration-friendly literature pipeline.

The goal of this module is to keep all cross-module contracts in one place so
you can copy the search / LaTeX / citation-review modules into another agent
framework with minimal changes.

中文说明：
- 本模块定义了在检索、引用核验与 LaTeX 转换模块之间共享的数据结构和
 轻量工具函数，保证在迁移到其他 agent 框架时减少耦合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import re


# 一个非常小的停用词列表，足以用于 cite-key 生成和查询简化。
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(slots=True)
class Author:
    """Author data kept intentionally small for easy serialization."""

    # 仅保存姓名，避免复杂作者元数据导致跨环境序列化问题。
    name: str

    def last_name(self) -> str:
        """返回用于生成 BibTeX 键的“姓氏”部分（已清洗）。

        实现策略：按空白拆分名字，取最后一段作为家族名；再去除非字母数字字符并
        转小写；若无有效部分则返回 "anon"。
        """
        parts = [part for part in re.split(r"\s+", self.name.strip()) if part]
        last = parts[-1] if parts else "anon"
        return re.sub(r"[^A-Za-z0-9]+", "", last).lower() or "anon"


@dataclass(slots=True)
class Paper:
    """Normalized paper object shared across search and citation-review code."""
    # 论文的规范化表示：尽量只包含常见且稳定的字段，方便序列化/缓存/去重。
    title: str
    authors: list[Author] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    arxiv_id: str = ""
    semantic_scholar_id: str = ""
    openalex_id: str = ""
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cite_key(self) -> str:
        """Create a stable BibTeX key similar to the original repo."""
        # 优先用第一作者姓氏，和学术写作里常见的 BibTeX key 习惯一致。
        family = self.authors[0].last_name() if self.authors else "anon"
        # 年份缺失时用 nodate，避免生成空 key。
        year = str(self.year or "nodate")
        words = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", self.title)
            if token.lower() not in _STOPWORDS
        ]
        # 只取前两个实词，保证 key 短而稳定。
        stem = "".join(words[:2]) or "paper"
        return f"{family}{year}{stem}"

    def to_bibtex(self) -> str:
        """Render a minimal BibTeX entry that compiles in standard workflows."""
        # 先清掉可能干扰 BibTeX 结构的花括号，保证输出不会破坏 .bib 格式。
        safe_title = self.title.replace("{", "").replace("}", "").strip()
        safe_venue = self.venue.replace("{", "").replace("}", "").strip()
        author_text = " and ".join(author.name for author in self.authors) or "Unknown"
        fields = [
            f"  title = {{{safe_title}}}",
            f"  author = {{{author_text}}}",
        ]
        if self.year:
            fields.append(f"  year = {{{self.year}}}")
        if safe_venue:
            fields.append(f"  journal = {{{safe_venue}}}")
        if self.doi:
            fields.append(f"  doi = {{{self.doi}}}")
        if self.url:
            fields.append(f"  url = {{{self.url}}}")
        return "@article{" + self.cite_key + ",\n" + ",\n".join(fields) + "\n}\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for JSONL or cache writes."""
        return {
            "title": self.title,
            "authors": [author.name for author in self.authors],
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
            "semantic_scholar_id": self.semantic_scholar_id,
            "openalex_id": self.openalex_id,
            "source": self.source,
            "cite_key": self.cite_key,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Rebuild a normalized paper object from cached JSON data."""
        authors = [Author(name=name) for name in data.get("authors", []) if isinstance(name, str)]
        return cls(
            title=str(data.get("title", "")).strip(),
            authors=authors,
            abstract=str(data.get("abstract", "") or ""),
            year=data.get("year"),
            venue=str(data.get("venue", "") or ""),
            doi=str(data.get("doi", "") or ""),
            url=str(data.get("url", "") or ""),
            pdf_url=str(data.get("pdf_url", "") or ""),
            arxiv_id=str(data.get("arxiv_id", "") or ""),
            semantic_scholar_id=str(data.get("semantic_scholar_id", "") or ""),
            openalex_id=str(data.get("openalex_id", "") or ""),
            source=str(data.get("source", "") or ""),
            raw=data.get("raw", {}) if isinstance(data.get("raw"), dict) else {},
        )


class CitationStatus(str, Enum):
    # 引用验证状态枚举：表示条目的核验结果（已验证 / 可疑 / 幻觉 / 跳过）。
    """Tri-state plus skipped status used by the verification report."""

    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    HALLUCINATED = "hallucinated"
    SKIPPED = "skipped"


@dataclass(slots=True)
class CitationResult:
    # 单条 BibTeX 条目的核验结果结构，供验证报告和输出使用。
    """One verification result for one BibTeX entry."""

    key: str
    title: str
    status: CitationStatus
    reason: str
    matched_title: str = ""
    matched_doi: str = ""
    source: str = ""
    score: float = 0.0


@dataclass(slots=True)
class VerificationReport:
    """Aggregated citation verification output."""

    results: list[CitationResult]

    @property
    def verified(self) -> int:
        return sum(item.status == CitationStatus.VERIFIED for item in self.results)

    @property
    def suspicious(self) -> int:
        return sum(item.status == CitationStatus.SUSPICIOUS for item in self.results)

    @property
    def hallucinated(self) -> int:
        return sum(item.status == CitationStatus.HALLUCINATED for item in self.results)

    @property
    def skipped(self) -> int:
        return sum(item.status == CitationStatus.SKIPPED for item in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for JSON writes or agent messaging."""
        return {
            "verified": self.verified,
            "suspicious": self.suspicious,
            "hallucinated": self.hallucinated,
            "skipped": self.skipped,
            "results": [
                {
                    "key": item.key,
                    "title": item.title,
                    "status": item.status.value,
                    "reason": item.reason,
                    "matched_title": item.matched_title,
                    "matched_doi": item.matched_doi,
                    "source": item.source,
                    "score": item.score,
                }
                for item in self.results
            ],
        }


@dataclass(slots=True)
class ConferenceTemplate:
    # LaTeX 模板的元数据（文档类、选项、参考文献样式等），与具体实现解耦。
    """LaTeX template metadata kept independent from the original repo."""

    name: str
    document_class: str
    class_options: str = ""
    bibliography_style: str = "plain"
    use_natbib: bool = True
    extra_preamble: str = ""
    style_files: list[Path] = field(default_factory=list)

    def get_style_files(self) -> list[Path]:
        """Return bundled style files that should be copied near paper.tex."""
        return list(self.style_files)


@dataclass(slots=True)
class CompileResult:
    # LaTeX 编译过程的结果封装，包含是否成功、生成的 PDF/日志路径与错误摘要。
    """Result of a LaTeX compilation attempt."""

    success: bool
    pdf_path: Path | None
    log_path: Path | None
    commands: list[list[str]]
    errors: list[str]
