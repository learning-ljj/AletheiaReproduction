# Migration Reference

This directory contains two low-coupling reference modules extracted from the
core ideas of this repository.

- `literature_pipeline.py`
  A small literature service that focuses on:
  - stable paper search with fallback and retry
  - PDF source resolution
  - PDF text extraction
  - extractive summarization

- `research_workflow_engine.py`
  A small research workflow engine that focuses on:
  - persistent project/workflow state
  - task-driven workflow progression
  - claim / evidence / artifact linkage
  - experiment contract validation and remediation hints

The code is intentionally independent from the current repo runtime so you can
copy it into your own agent framework with minimal changes.

## Minimal Usage

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

## Why These Modules Exist

The original repository has two different layers:

- a full runtime layer: CLI, FastAPI, multi-agent runner, scheduler
- a research state layer: project, workflow, note, claim, evidence, experiment

For migration learning, the runtime layer is often too heavy.
These reference modules keep the business logic and remove the framework glue.

核心文件：

literature_pipeline.py
从 SemanticScholarSearchBackend (line 269)、ArxivSearchBackend (line 331)、StableLiteratureService (line 397) 开始，拆出了“稳定检索”；PdfExtractor (line 483) 负责本地文件 / URL / arXiv id 的 PDF 解析；ExtractivePaperSummarizer (line 722) 负责无 LLM 的可迁移摘要和 claim 候选生成。
research_workflow_engine.py
JsonStateStore (line 402) 负责长期状态持久化；ResearchWorkflowService (line 445) 负责 project/workflow/task/note/artifact/claim/evidence/experiment 的状态变更；update_experiment_result (line 1055)、claim_graph (line 1140)、dashboard (line 1168)、record_literature_search (line 1214)、record_paper_summary (line 1291) 对应你最关心的链路；ResearchWorkflowRuntime (line 1398) 把“状态机”和“你的 agent 执行器”解耦了。
README.md
给了最小接线示例。