# 迁移核心（Migration Core）

本目录将原仓库中最适合迁移的三段能力拆解为低耦合、易集成的实现，便于把功能迁入你自己的 agent 框架中。

主要文件说明：

- `stable_literature_search.py`：稳定的文献检索模块，包含多源检索、查询扩展、缓存回退、限流、简易熔断以及 DOI/arXiv/标题去重逻辑。
- `latex_pipeline.py`：负责把 Markdown 转成 LaTeX（`paper.tex`），并可选调用 `pdflatex` / `bibtex` 进行编译生成 PDF。
- `citation_review.py`：引用核验与清洗，按照 DOI -> OpenAlex -> arXiv -> 标题 的核验顺序执行，并包含 BibTeX 过滤与正文坏引用（hallucination）清理逻辑。
- `shared_models.py`：共享数据结构定义，降低对原仓库 stage/config 系统的耦合。
- `__init__.py`：包入口。

设计原则：

- 仅依赖 Python 标准库与本目录内的 `shared_models.py`，尽量降低外部依赖。
- 不依赖原仓库的 stage/runner/配置系统，便于直接在任意 agent 框架中复用。
- 把可能的 LLM 复核路径设计为可选回调（callback），不强制依赖特定模型 SDK。
- 代码中保留了逐行注释，便于迁移时逐步理解实现细节。

主要用法示例

1) 稳定文献检索（CLI 示例）：

```bash
python -m docs.analysis.migration_core.stable_literature_search \
  --topic "retrieval augmented generation for scientific literature review" \
  --query "RAG literature review" \
  --jsonl-out candidates.jsonl \
  --bib-out references.bib
```

在 agent 中更常见的用法是直接 import：

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

2) LaTeX 生成与渲染：

```bash
python -m docs.analysis.migration_core.latex_pipeline \
  --markdown paper.md \
  --out-dir out_latex \
  --template generic \
  --author "Alice Zhang" \
  --author "Bob Li" \
  --compile
```

如果只需要生成 `paper.tex`，可省略 `--compile`。

3) 文献引用审查：

```bash
python -m docs.analysis.migration_core.citation_review \
  --bib references.bib \
  --out-bib references_verified.bib \
  --report-json verification_report.json \
  --paper-md paper.md \
  --paper-out paper_verified.md
```

与原仓库的主要差异

- 检索模块保留 API 检索与缓存后备，但去掉了依赖 LLM 的 fallback stage 与产物目录约束，便于在不同运行时直接复用。
- LaTeX 模块保留模板/样式复制与编译链，但只内置了轻量模板（如 `generic` 与 `neurips_2025`），以减少迁移成本。
- 引用核验保留 DOI -> OpenAlex -> arXiv -> 标题 的核验顺序，同时将“主题相关性复核”实现为可选回调，使用者可接入自有的模型或规则。

推荐迁移顺序

1. 优先接入 `stable_literature_search.py`（独立、复用成本低）。
2. 接入 `citation_review.py`，对检索得到的 BibTeX 做真实性与格式过滤。
3. 最后接 `latex_pipeline.py`，将 agent 产出的 Markdown 导出为 LaTeX/PDF。

快速文件索引（便于阅读实现）：

- 稳定检索实现：`stable_literature_search.py`
- 引用核验实现：`citation_review.py`
- LaTeX 生成：`latex_pipeline.py`
- 共享模型：`shared_models.py`

如果你希望，我可以把每个模块的用法示例展开为更详细的迁移步骤和调用样例。 
