"""工作日志构建器：从 raw JSONL 离线生成结构化 Markdown。"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging.raw_log_reader import load_raw_events
from src.utils.logging.worklog_components import (
    EventCollector,
    MarkdownRenderer,
    RoleSummarizer,
    SummaryLLMService,
)


class WorklogBuilder:
    """Worklog facade that composes collector/summarizer/renderer components."""

    def __init__(self, llm_client=None, llm_config: dict | None = None, prompts: dict | None = None):
        self.summary_service = SummaryLLMService(
            llm_client=llm_client,
            llm_config=llm_config,
            prompts=prompts,
        )
        self.collector = EventCollector()
        self.summarizer = RoleSummarizer(self.summary_service)
        self.renderer = MarkdownRenderer(self.summarizer)

    def build_problem_worklog(self, run_jsonl_path: str, output_md_path: str) -> None:
        self.summary_service.begin_session()
        try:
            run_path = Path(run_jsonl_path)
            output_path = Path(output_md_path)

            if run_path.name == "history.jsonl" and run_path.parent.name:
                problem_id = run_path.parent.name
                runs_root = run_path.parent.parent
            else:
                problem_id = run_path.stem
                runs_root = run_path.parent

            events = load_raw_events(problem_id=problem_id, runs_root=runs_root)
            run_data = self.collector.collect(problem_id=problem_id, events=events)
            markdown = self.renderer.render(run_data)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
        finally:
            self.summary_service.end_session()
