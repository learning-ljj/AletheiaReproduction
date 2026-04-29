"""Per-problem memory manager for durable state, history, and artifacts.

提供基于问题 ID 的持久化存储，管理状态快照、历史事件以及各类工件（引理、论文、错误记录、引用等）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.memory.state import ProblemSnapshot

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
        """从 YAML frontmatter 提取 title 字段，若没有则尝试提取 summary 字段。"""
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
            key_lower = key.strip().lower()
            if key_lower == "title" or key_lower == "summary":
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
        """将标准三层 Markdown 引理内容持久化到 lemmas 目录。

        内容应为以下三种格式之一：
        1. 完整三层 Markdown（推荐，含 YAML 头部）。
        2. 仅有 YAML 头部与证明。
        3. 纯文本（无 YAML 头部），此时自动生成 title 及默认元数据。

        文件名根据 YAML 中的 title 字段生成，若缺失则尝试用第一行非空文本，
        若仍无法提取有意义文字，则使用内容哈希作为文件名（以保证稳定性）。
        若文件名冲突（相同 title 但不同内容），自动追加 _2, _3 等编号并记录警告。
        内容完全相同的引理不会被重复写入。
        """
        self.init_dirs()

        # 1) 去重：若已有完全相同内容的文件，直接复用
        existing = self._find_existing_markdown_by_content(self.lemmas_dir, content)
        if existing is not None:
            return existing

        # 2) 确定一个有意义且安全的文件名
        title = self._title_from_frontmatter(content)
        if not title:
            title = self._first_non_empty_line(content)

        desired_name = self._slugify_filename(title) if title else None
        if not desired_name:
            # 无法提取任何可读文字时，用内容哈希生成稳定文件名
            hash_hex = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
            desired_name = f"lemma_{hash_hex}.md"
        else:
            desired_name = f"{desired_name}.md"

        # 3) 处理文件名冲突（相同文件名但不同内容）
        final_name = self._resolve_filename_collision_in_dir(
            self.lemmas_dir, desired_name
        )

        # 4) 写入
        target = self._save_markdown(self.lemmas_dir, content, filename=final_name)

        # 5) 记录新增（如果没有被去重跳过）
        if target not in self._initial_lemma_paths:
            self._new_lemma_paths.add(target)

        return target

    @staticmethod
    def _slugify_filename(text: str, max_len: int = 60) -> str:
        """将文本转换为安全的文件名片段。"""
        # 保留字母、数字、空白、连字符；去除其余字符
        slug = re.sub(r'[^\w\s-]', '', text.lower()).strip()
        # 空白和连字符统一替换为下划线
        slug = re.sub(r'[-\s]+', '_', slug)
        # 去除首尾下划线，截断
        return slug[:max_len].strip('_')

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
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                # 优先 frontmatter title，其次首行
                title = (
                    self._title_from_frontmatter(text)
                    or self._first_non_empty_line(text)
                )
                if not title:
                    continue
                relative = path.relative_to(self.run_dir).as_posix()
                out.append(f"{title} [path:{relative}]")
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

    def save_bib_entries(self, entries: list[str]) -> Path:
        return self.save_bibtex(
            "\n\n".join((entry or "").strip() for entry in entries if entry)
        )

    def save_manifest(self, payload: dict[str, Any]) -> Path:
        if not isinstance(payload, dict):
            raise TypeError("manifest payload must be a dict")
        self.init_dirs()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write_text(self.manifest_path, serialized + "\n")
        return self.manifest_path