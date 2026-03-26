# Migration Core

stable_literature_search.py：稳定文献检索。保留了多源检索、查询扩展、缓存回退、限流、简易熔断、DOI/arXiv/title 去重。
latex_pipeline.py：LaTeX 生成与渲染。保留了 Markdown 转 paper.tex、模板选择、样式复制、pdflatex/bibtex 编译。
citation_review.py：文献引用审查。保留了 DOI -> OpenAlex -> arXiv -> 标题搜索的核验顺序，以及 BibTeX 过滤、正文坏引用清理。
shared_models.py：共享数据结构，避免你迁移时被原仓库的 stage/config 体系绑住。
init.py：包入口。
README.md：最小使用方式和迁移说明。

这个目录把原仓库里最适合迁移的 3 段能力拆成了低耦合版本：

- `stable_literature_search.py`
  负责稳定文献检索，保留了多源查询、缓存回退、限流和 DOI/arXiv/title 去重。
- `latex_pipeline.py`
  负责 Markdown 转 `paper.tex`，并可选调用 `pdflatex` / `bibtex` 生成 PDF。
- `citation_review.py`
  负责引用真实性核验、BibTeX 过滤、正文幻觉引用清理。

## 设计原则

- 只依赖 Python 标准库和本目录下的 `shared_models.py`。
- 不依赖 ResearchClaw 的 stage、runner、配置系统。
- 保留原仓库最关键的稳定性机制，但把 LLM 依赖降成可选回调。
- 所有关键逻辑都加了紧贴代码的逐行注释，方便你边读边迁。

## 1. 稳定文献检索

```bash
python -m docs.analysis.migration_core.stable_literature_search ^
  --topic "retrieval augmented generation for scientific literature review" ^
  --query "RAG literature review" ^
  --jsonl-out candidates.jsonl ^
  --bib-out references.bib
```

你在 agent 框架里更常见的接法是直接 import：

```python
from docs.analysis.migration_core.stable_literature_search import SearchConfig, search_papers_multi_query

config = SearchConfig(
    sources=["openalex", "semantic_scholar", "arxiv"],
    limit_per_query=10,
    year_min=2021,
    semantic_scholar_api_key=None,
)
papers = search_papers_multi_query(
    ["RAG literature review"],
    topic="retrieval augmented generation for scientific literature review",
    config=config,
)
```

## 2. LaTeX 生成与渲染

```bash
python -m docs.analysis.migration_core.latex_pipeline ^
  --markdown paper.md ^
  --out-dir out_latex ^
  --template generic ^
  --author "Alice Zhang" ^
  --author "Bob Li" ^
  --compile
```

如果你只想拿 `paper.tex`，不需要 `--compile`。

## 3. 文献引用审查

```bash
python -m docs.analysis.migration_core.citation_review ^
  --bib references.bib ^
  --out-bib references_verified.bib ^
  --report-json verification_report.json ^
  --paper-md paper.md ^
  --paper-out paper_verified.md
```

## 与原仓库的差异

- 检索模块保留了真实 API 搜索和缓存回退，但去掉了 LLM fallback 和 stage 产物目录约束。
- LaTeX 模块保留了模板选择和编译链，但只内置了 `generic` 和 `neurips_2025` 两个轻量模板。
- 引用核验保留了 DOI -> OpenAlex -> arXiv -> 标题搜索顺序，但把“主题相关性复核”做成了可选回调，不强绑某个模型 SDK。

## 推荐迁移顺序

1. 先接 `stable_literature_search.py`，因为它最独立。
2. 再接 `citation_review.py`，把检索结果的 BibTeX 做真实性过滤。
3. 最后接 `latex_pipeline.py`，把你自己 agent 生成的 Markdown 导出成 LaTeX/PDF。
