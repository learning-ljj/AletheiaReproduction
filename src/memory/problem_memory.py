"""Per-problem memory manager for durable state, history, and artifacts.

提供基于问题 ID 的持久化存储，管理状态快照、历史事件以及各类工件（引理、论文、错误记录、引用等）。
"""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.memory.state import ProblemSnapshot, StageSnapshot, EventSnapshot
from src.tools.artifact_reader import read_artifact

_logger = logging.getLogger(__name__)

# ── 上下文变量：在当前协程/线程中隐式传递 ProblemMemory 实例 ──
_CURRENT_PROBLEM_MEMORY: ContextVar["ProblemMemory | None"] = ContextVar(
    "current_problem_memory",
    default=None,
)

def set_current_problem_memory(memory: "ProblemMemory | None") -> None:
    _CURRENT_PROBLEM_MEMORY.set(memory)

def get_current_problem_memory() -> "ProblemMemory | None":
    return _CURRENT_PROBLEM_MEMORY.get()

# ── 核心类 ──
class ProblemMemory:
    """Manage persistence under runs/{problem_id}.

    目录布局:
      runs/{problem_id}/
        state.json
        history.jsonl
        artifact/
          lemmas/
          papers/
          errors/
          citations.bib
    """

    def __init__(self, problem_id: str, runs_root: Path | str = "runs"):
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")

        self.problem_id = problem_id.strip()
        self.runs_root = Path(runs_root)
        self.run_dir = self.runs_root / self.problem_id
        self.state_path = self.run_dir / "state.json"
        self.history_path = self.run_dir / "history.jsonl"

        self.artifact_dir = self.run_dir / "artifact"
        self.lemmas_dir = self.artifact_dir / "lemmas"
        self.papers_dir = self.artifact_dir / "papers"
        self.errors_dir = self.artifact_dir / "errors"
        self.citations_bib_path = self.artifact_dir / "citations.bib"
        self.manifest_path = self.artifact_dir / "manifest.json"

        # 本次运行初始已存在的引理文件集合，用于判断“新增”
        self._initial_lemma_paths: set[Path] = set()
        self._new_lemma_paths: set[Path] = set()

        self.init_dirs()

    # ═════════════════════════════════════════════════════════════
    # 块 1：初始化与目录管理
    # ═════════════════════════════════════════════════════════════
    def init_dirs(self) -> None:
        """确保所有必需目录存在（幂等）。首次调用时记录初始引理文件列表。"""
        self.lemmas_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

        if not self._initial_lemma_paths:
            self._initial_lemma_paths = set(self.lemmas_dir.glob("*.md"))

    # ═════════════════════════════════════════════════════════════
    # 块 2：底层原子写入工具
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """原子写入：先写临时文件再重命名，防止写坏。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)

    # ═════════════════════════════════════════════════════════════
    # 块 3：状态快照持久化
    # ═════════════════════════════════════════════════════════════
    def save_state(self, state: ProblemSnapshot | dict[str, Any]) -> ProblemSnapshot:
        self.init_dirs()
        snapshot = (
            state
            if isinstance(state, ProblemSnapshot)
            else ProblemSnapshot.from_dict(state)
        )
        serialized = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        self._atomic_write_text(self.state_path, serialized + "\n")
        return snapshot

    def load_state(self) -> ProblemSnapshot | None:
        if not self.state_path.exists():
            return None
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return ProblemSnapshot.from_dict(data)

    def merge_state(self, patch: dict[str, Any]) -> ProblemSnapshot:
        if not isinstance(patch, dict):
            raise TypeError("patch must be a dict")
        current = self.load_state()
        merged = current.to_dict() if current else {}
        merged.update(patch)
        return self.save_state(ProblemSnapshot.from_dict(merged))

    def record_stage_event(
        self,
        stage_name: str,
        turn_id: int,
        status: str,
        detail: str | None = None,
        error: str | None = None,
        timestamp: str | None = None,
        event_detail: dict[str, Any] | None = None,
    ) -> None:
        """记录单个阶段的执行事件。
        
        参数:
        - stage_name: 阶段名（例如 'GENERATOR', 'VERIFIER', 'REVISER'）
        - turn_id: 所在的迭代轮次
        - status: 该阶段的执行状态（例如 'SUCCESS', 'FAILED'）
        - detail: 阶段结果的简短摘要（可选）
        - error: 错误信息（若有）
        - timestamp: 事件时间戳（默认使用当前UTC时间）
        - event_detail: 附加细节字典（工具调用、参数等）
        
        流程:
        1. 读取当前state.json
        2. 在stages数组中找或创建该turn_id对应的StageSnapshot
        3. 添加事件到该阶段的events数组
        4. 保存更新后的state
        """
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string")
        if not isinstance(turn_id, int) or turn_id < 0:
            raise ValueError("turn_id must be a non-negative integer")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")

        # 默认使用当前UTC时间
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # 读取当前状态
        current_state = self.load_state()
        if current_state is None:
            # 如果还没有state.json，创建一个默认的
            current_state = ProblemSnapshot(
                problem_id=self.problem_id,
                iteration_count=turn_id,
                status="RUNNING",
            )

        # 查找对应turn_id的stage，或创建新的
        stage_snapshot = None
        for stage in current_state.stages:
            if stage.turn_id == turn_id and stage.stage_name == stage_name:
                stage_snapshot = stage
                break

        if stage_snapshot is None:
            stage_snapshot = StageSnapshot(
                stage_name=stage_name,
                turn_id=turn_id,
                status=status,
                detail=detail,
                last_error=error,
                events=[],
            )
            current_state.stages.append(stage_snapshot)
        else:
            # 更新阶段的状态信息
            stage_snapshot.status = status
            if detail is not None:
                stage_snapshot.detail = detail
            if error is not None:
                stage_snapshot.last_error = error

        # 创建事件对象并添加到该阶段
        event = EventSnapshot(
            event_type="EXECUTION",
            status=status,
            timestamp=timestamp,
            error=error,
            detail=event_detail,
        )
        stage_snapshot.events.append(event)

        # 保存更新后的状态
        self.save_state(current_state)

    # ═════════════════════════════════════════════════════════════
    # 块 4：历史事件（追加型 JSONL）
    # ═════════════════════════════════════════════════════════════
    def append_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        required = ("node", "turn_id", "timestamp")
        missing = [k for k in required if k not in event]
        if missing:
            raise ValueError(f"event missing required keys: {missing}")

        self.init_dirs()
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with open(self.history_path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.history_path} at line {line_no}: {exc.msg}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Invalid event object in {self.history_path} at line {line_no}: not a JSON object"
                    )
                events.append(obj)
        return events

    # ═════════════════════════════════════════════════════════════
    # 块 5：Markdown 工件通用写入
    # ═════════════════════════════════════════════════════════════
    def _save_markdown(self, folder: Path, content: str, filename: str) -> Path:
        """向指定目录写入 Markdown 文件。如果目标文件已存在且内容一致，直接复用。"""
        self.init_dirs()
        target = folder / filename
        normalized = (content or "").rstrip() + "\n"

        if target.exists():
            if target.read_text(encoding="utf-8") == normalized:
                return target

        self._atomic_write_text(target, normalized)
        return target

    def _find_existing_markdown_by_content(self, folder: Path, content: str) -> Path | None:
        """在目录中按内容查找完全相同的 `.md` 文件，找到返回路径，否则返回 None。"""
        normalized = (content or "").rstrip() + "\n"
        for path in sorted(folder.glob("*.md")):
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if existing == normalized:
                return path
        return None

    # ═════════════════════════════════════════════════════════════
    # 块 6：Lemma（引理）管理
    # ═════════════════════════════════════════════════════════════
    def initial_lemma_count(self) -> int:
        """返回本次运行开始之前已存在的引理文件数量。"""
        return len(self._initial_lemma_paths)

    @staticmethod
    def _title_from_frontmatter(text: str) -> str | None:
        """从 YAML frontmatter 提取 title 字段。"""
        stripped = (text or "").lstrip()
        if not stripped.startswith("---"):
            return None

        lines = stripped.splitlines()
        if not lines or lines[0].strip() != "---":
            return None

        for line in lines[1:]:
            line = line.strip()
            if line == "---":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "title":
                return value.strip()
        return None

    @staticmethod
    def _first_non_empty_line(text: str) -> str | None:
        """返回文本第一个非空行。"""
        for line in (text or "").splitlines():
            value = line.strip()
            if value:
                return value
        return None

    def add_lemma(self, content: str) -> Path:
        """将引理内容持久化到 lemmas 目录。

        规则：
        - 内容完全相同则复用已有文件；
        - 文件名只能来自 YAML frontmatter 的 title；
        - 缺少 title 直接报错；
        - 同名但不同内容时追加编号并记录警告。
        """
        self.init_dirs()

        normalized_content = self._normalize_lemma_content(content)

        # 1) 去重：若已有完全相同内容的文件，直接复用
        existing = self._find_existing_markdown_by_content(self.lemmas_dir, normalized_content)
        if existing is not None:
            return existing

        # 2) 文件名必须来自 frontmatter 的 title，缺失就直接报错
        title = self._title_from_frontmatter(normalized_content)
        if not title:
            raise ValueError("lemma missing YAML frontmatter title")

        desired_name = self._slugify_filename(title)
        if not desired_name:
            raise ValueError(f"invalid lemma title for filename: {title!r}")
        desired_name = f"{desired_name}.md"

        # 3) 处理文件名冲突（相同文件名但不同内容）
        final_name = self._resolve_filename_collision_in_dir(
            self.lemmas_dir, desired_name
        )

        run_path = f"runs/{self.problem_id}/artifact/lemmas/{final_name}"
        normalized_content = self._replace_lemma_source(
            normalized_content,
            from_path=run_path,
        )

        # 4) 写入
        target = self._save_markdown(self.lemmas_dir, normalized_content, filename=final_name)

        # 5) 记录新增（如果没有被去重跳过）
        if target not in self._initial_lemma_paths:
            self._new_lemma_paths.add(target)

        return target

    @staticmethod
    def _replace_lemma_source(text: str, from_path: str) -> str:
        """将 frontmatter 中的 source: self_proved 替换为 from: <path>。
        
        替换方式避免 re.sub() 中的转义问题，改用 callable 替换或直接字符串拼接。
        """
        if not text:
            return text
        pattern = re.compile(r"(?s)\A(\s*---\n)(.*?)(\n---)")
        match = pattern.search(text)
        if not match:
            return text

        header, body, tail = match.groups()
        lines = body.splitlines()
        replaced = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*source\s*:\s*self_proved\s*$", line):
                lines[i] = f"from: {from_path}"
                replaced = True
                break
        if not replaced:
            lines.append(f"from: {from_path}")
        new_body = "\n".join(lines)
        # 使用直接字符串拼接而不是 pattern.sub()，避免转义问题
        frontmatter = f"{header}{new_body}{tail}"
        rest = text[match.end():]
        return frontmatter + rest

    @staticmethod
    def _slugify_filename(text: str, max_len: int = 60) -> str:
        # 删除 Windows 绝对禁止的字符，其余字符（包括中文）全部保留。
        text = re.sub(r'[\\/:*?"<>|]', '', text)
        # 压缩连续空白，避免标题里过多换行或制表符进入文件名。
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除首尾的空格、点号、连字符或下划线，避免 Windows 文件名边界问题。
        text = text.strip('. _-')
        # 限制长度并返回。
        return text[:max_len] or 'lemma'

    @staticmethod
    def _normalize_lemma_content(content: str) -> str:
        r"""规范化 lemma 内容中的展示数学格式，把文本中的 LaTeX 显示数学公式标记 \[ ... \] 转换成 Markdown/数学渲染器更通用的 $$ ... $$ 格式。"""
        normalized = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.S)
        return normalized

    @staticmethod
    def _resolve_filename_collision_in_dir(directory: Path, desired_name: str) -> str:
        """检查文件名冲突，若冲突则追加编号（如 _2.md），并记录警告。"""
        existing_names = set(p.name for p in directory.glob("*.md"))
        candidate = desired_name
        if candidate not in existing_names:
            return candidate

        # 分离文件名与扩展名
        stem, ext = (
            candidate.rsplit('.', 1)
            if '.' in candidate
            else (candidate, 'md')
        )
        # 如果已经以 _数字 结尾，去掉该数字后缀
        base = re.sub(r'_(\d+)$', '', stem)

        counter = 2
        while f"{base}_{counter}.{ext}" in existing_names:
            counter += 1
        candidate = f"{base}_{counter}.{ext}"

        # 记录冲突警告
        _logger.warning(
            "Lemma 文件名冲突：'%s' 已存在，将使用 '%s'",
            desired_name,
            candidate,
        )
        return candidate

    def count_lemmas(self) -> int:
        """返回当前 lemmas 目录下 .md 文件总数。"""
        self.init_dirs()
        return sum(1 for _ in self.lemmas_dir.glob("*.md"))

    def count_new_lemmas_since_start(self) -> int:
        """返回本次运行开始以来实际新增的引理文件数量。"""
        return len(self._new_lemma_paths)

    def list_lemma_context_items(self, limit: int = 12) -> list[str]:
        """获取引理/论文的轻量级摘要列表，用于上下文注入。"""
        self.init_dirs()
        out: list[str] = []
        for folder in (self.lemmas_dir, self.papers_dir):
            for path in sorted(folder.glob("*.md")):
                run_path = f"runs/{self.problem_id}/{path.relative_to(self.run_dir).as_posix()}"
                payload = read_artifact(path=run_path, layer=1)
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if parsed.get("status") != "OK":
                    continue
                data = parsed.get("data") or {}
                content = data.get("content")
                if not content:
                    continue
                out.append(content)
                if len(out) >= max(0, limit):
                    return out
        return out

    # ═════════════════════════════════════════════════════════════
    # 块 7：论文与错误工件
    # ═════════════════════════════════════════════════════════════
    def add_paper(self, content: str, filename: str | None = None) -> Path:
        """添加一篇论文工件。"""
        # 若未提供文件名，用同样的 title 提取 + 冲突处理逻辑
        if filename is None:
            title = self._title_from_frontmatter(content) or self._first_non_empty_line(content)
            if title:
                base = self._slugify_filename(title)
                if base:
                    desired = f"{base}.md"
                    filename = self._resolve_filename_collision_in_dir(self.papers_dir, desired)
        if filename is None:
            filename = f"paper_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
        return self._save_markdown(self.papers_dir, content, filename=filename)

    def add_error(self, content: str, filename: str | None = None) -> Path:
        """添加一条错误记录工件。"""
        if filename is None:
            filename = f"error_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
        return self._save_markdown(self.errors_dir, content, filename=filename)

    # ═════════════════════════════════════════════════════════════
    # 块 8：BibTeX 引用与运行清单
    # ═════════════════════════════════════════════════════════════
    def save_bibtex(self, bibtex: str) -> Path:
        self.init_dirs()
        normalized = (bibtex or "").rstrip() + "\n"
        if self.citations_bib_path.exists():
            if self.citations_bib_path.read_text(encoding="utf-8") == normalized:
                return self.citations_bib_path
        self._atomic_write_text(self.citations_bib_path, normalized)
        return self.citations_bib_path

    # def save_bib_entries(self, entries: list[str]) -> Path:
    #     return self.save_bibtex(
    #         "\n\n".join((entry or "").strip() for entry in entries if entry)
    #     )

    def save_manifest(self, payload: dict[str, Any]) -> Path:
        if not isinstance(payload, dict):
            raise TypeError("manifest payload must be a dict")
        self.init_dirs()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write_text(self.manifest_path, serialized + "\n")
        return self.manifest_path