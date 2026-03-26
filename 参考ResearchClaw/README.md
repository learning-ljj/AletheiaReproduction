# 迁移参考（Migration Reference）

本目录包含从原仓库抽出的两个低耦合参考模块，旨在保留业务逻辑，去除繁重的运行时框架，便于在你自己的 agent 框架中逐步迁移与集成。

包含模块：

- `literature_pipeline.py`：小型文献服务，侧重于稳定检索、PDF 源解析、PDF 文本抽取与抽取式摘要（extractive summarization）。
- `research_workflow_engine.py`：轻量研究工作流引擎，负责持久化项目/工作流状态、任务驱动的工作流推进，以及声明（claim）/证据（evidence）/产物（artifact）之间的关联管理。

该实现与当前仓库运行时独立，可直接拷贝到你的项目中并最小化改动。

最小使用示例：

```python
from pathlib import Path

from examples.migration_reference.literature_pipeline import (
    ArxivSearchBackend,
    ExtractivePaperSummarizer,
    PdfExtractor,
    SemanticScholarSearchBackend,
    StableLiteratureService,
)
from examples.migration_reference.research_workflow_engine import (
    JsonStateStore,
    ResearchWorkflowService,
)

literature = StableLiteratureService(
    backends=[
        SemanticScholarSearchBackend(),
        ArxivSearchBackend(),
    ],
)
extractor = PdfExtractor(download_dir=Path("./tmp_papers"))
summarizer = ExtractivePaperSummarizer()

store = JsonStateStore(Path("./tmp_state/research_state.json"))
service = ResearchWorkflowService(store)

project = service.create_project(name="Distribution Shift Project")
workflow = service.create_workflow(
    project_id=project.id,
    title="Robustness Survey",
    goal="Find stable evidence for robustness under distribution shift.",
)

search_result = literature.search(
    query="distribution shift robustness benchmark",
    max_results=5,
)

service.record_literature_search(
    workflow_id=workflow.id,
    query=search_result.query,
    source=search_result.used_backend,
    papers=[paper.to_dict() for paper in search_result.papers],
)

paper = search_result.papers[0]
pdf = extractor.extract(source=paper.best_pdf_url or paper.url)
summary = summarizer.summarize(
    title=paper.title,
    text=pdf.text,
    level="standard",
)
service.record_paper_summary(
    workflow_id=workflow.id,
    paper=paper.to_dict(),
    summary=summary.to_dict(),
    claims=[
        "The paper defines a robustness evaluation setup that is reusable.",
        "The paper reports a meaningful performance drop under shift.",
    ],
)
```

为什么存在这些模块

原始仓库包含两套不同的层：

- 完整的运行时层：包含 CLI、FastAPI、multi-agent runner、调度器等。
- 研究状态层：project、workflow、note、claim、evidence、experiment 等状态模型与变更逻辑。

对于迁移学习场景，完整运行时往往过于沉重。因此这些参考模块把业务逻辑抽出，同时去除框架胶水部分，便于逐步接入。

核心实现文件（快速定位）

- `literature_pipeline.py`
  - 从 `SemanticScholarSearchBackend`、`ArxivSearchBackend`、`StableLiteratureService` 开始，拆出“稳定检索”能力；
  - `PdfExtractor` 负责本地文件/URL/arXiv id 的 PDF 下载与解析；
  - `ExtractivePaperSummarizer` 提供无 LLM 的可迁移摘要与 claim 候选生成。

- `research_workflow_engine.py`
  - `JsonStateStore`：简单的长期状态持久化实现；
  - `ResearchWorkflowService`：封装 project/workflow/task/note/artifact/claim/evidence/experiment 的状态变更接口；
  - 关键方法（例如 `update_experiment_result`、`claim_graph`、`dashboard`、`record_literature_search`、`record_paper_summary`）对应常见的链路操作；
  - `ResearchWorkflowRuntime` 将状态机与外部 agent 执行器解耦，便于在不同执行环境中复用。

README 提供了最小接线示例，便于快速运行与理解核心链路。

如果需要，我可以把 `literature_pipeline.py` 与 `research_workflow_engine.py` 的内部 API（参数与返回结构）整理成一份快速参考手册，方便在迁移时进行接口对接。