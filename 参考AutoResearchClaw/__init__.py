"""Migration-friendly extraction of ResearchClaw core capabilities."""

from docs.analysis.migration_core.citation_review import verify_citations
from docs.analysis.migration_core.latex_pipeline import compile_latex, get_template, markdown_to_latex
from docs.analysis.migration_core.shared_models import Author, Paper
from docs.analysis.migration_core.stable_literature_search import SearchConfig, search_papers_multi_query

__all__ = [
    "Author",
    "Paper",
    "SearchConfig",
    "search_papers_multi_query",
    "verify_citations",
    "get_template",
    "markdown_to_latex",
    "compile_latex",
]
