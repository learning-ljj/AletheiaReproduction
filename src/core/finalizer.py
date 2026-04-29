"""Final output helpers and terminal-state finalization engine.

负责将运行终态（SUCCESS、PROGRESS、FAILED）转化为最终输出文本，并持久化状态与工件。
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

from src.memory.state import ProofState, RunStatus, VerificationDecision, ProblemSnapshot

_logger = logging.getLogger(__name__)


class FinalizerEngine:
    """封装终态判定、最终文本构建和持久化副作用。

    所有持久化操作（事件日志、状态快照、Markdown 输出、清单）统一委托给 ProblemMemory。
    """

    def __init__(
        self,
        *,
        problem_memory,          # ProblemMemory 实例
        warnings: list[str],     # 运行过程中收集的所有警告
        runs_root: Path,
    ):
        self.problem_memory = problem_memory
        self.warnings = warnings
        self.runs_root = Path(runs_root)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _build_warning_summary(
        self, extra_warnings: list[str] | None = None
    ) -> str | None:
        """将全局警告与附加警告合并为 Markdown 列表，无警告时返回 None。"""
        merged = list(self.warnings)
        if extra_warnings:
            merged.extend(extra_warnings)
        if not merged:
            return None
        return "\n".join(f"- {msg}" for msg in merged)

    def _build_references_from_solution(
        self, solution_text: str | None
    ) -> tuple[str | None, list[str], list[str]]:
        """从解答文本抽取引用并导出 BibTeX。

        返回 (处理后文本, 引用条目列表, 引用相关警告)。
        """
        if self.problem_memory is None or not (solution_text or "").strip():
            return solution_text, [], []

        from src.utils.parsing.reference_builder import (
            build_references,
            export_references_bibtex,
        )

        try:
            converted, references, missing = build_references(
                solution_text or "", self.problem_memory
            )
            if references:
                try:
                    export_references_bibtex(references, self.problem_memory)
                except Exception as exc:
                    missing.append(
                        f"bibtex_export_error: {type(exc).__name__}: {exc}"
                    )
            return converted, references, missing
        except Exception as exc:
            return (
                solution_text,
                [],
                [f"reference_builder_error: {type(exc).__name__}: {exc}"],
            )

    def _build_final_output(
        self,
        *,
        success: bool,
        solution_text: str | None,
        failure_reason: str | None,
        progress: bool = False,
        references: list[str] | None = None,
        warning_summary: str | None = None,
    ) -> str:
        """构造最终输出文本（核心组合函数）。

        行为：
        - SUCCESS : 直接返回解答文本。
        - PROGRESS（轮次耗尽但有新增引理）：返回解答并附上状态说明。
        - FAILED：返回失败原因描述。
        - 最后追加引用和警告段落（如存在）。
        """
        reason = (failure_reason or "unknown_reason").strip()

        # 根据状态选择基础文本
        if success:
            base = (solution_text or "").strip()
        elif progress and solution_text:
            # 有进展但未完成：展示解答 + 状态提示
            base = (
                (solution_text or "").strip()
                + f"\n\nStatus: PROGRESS (max turns exhausted)."
            )
        else:
            base = f"Admits failure: {reason}."

        output = base.strip()

        if references:
            output += "\n\n## References\n" + "\n".join(references)

        if warning_summary:
            output += "\n\n## Citation Warnings\n" + warning_summary

        return output.strip()

    # ------------------------------------------------------------------
    # 统一组合入口
    # ------------------------------------------------------------------
    def _compose(
        self,
        state: ProofState,
        *,
        success: bool,
        failure_reason: str | None,
        progress: bool = False,
    ) -> tuple[str | None, list[str], str | None, str]:
        """处理引用、警告，生成最终输出文本。

        返回值：
            converted_solution, references, warning_summary, final_output
        """
        converted_solution, references, ref_warnings = (
            self._build_references_from_solution(state.current_proof)
        )
        warning_summary = self._build_warning_summary(ref_warnings)
        final_output = self._build_final_output(
            success=success,
            solution_text=converted_solution,
            failure_reason=failure_reason,
            progress=progress,
            references=references,
            warning_summary=warning_summary,
        )
        return converted_solution, references, warning_summary, final_output

    # ------------------------------------------------------------------
    # 持久化内部方法
    # ------------------------------------------------------------------
    def _persist(
        self,
        state: ProofState,
        *,
        turn_id: int,
        references: list[str],
        warning_summary: str | None,
        last_decision: VerificationDecision | None = None,
        extra_event: dict | None = None,
    ) -> None:
        """将终态写入事件日志、状态快照、输出工件和清单。"""
        if self.problem_memory is None:
            return

        # 1. 追加 FINAL 事件（核心字段由 state 提供，extra 补充细节）
        event_payload = {
            "node": "FINAL",
            "turn_id": turn_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": state.status.value if state.status else None,
            "failure_reason": state.failure_reason,
            "final_output": state.final_output,
        }
        if extra_event:
            event_payload.update(extra_event)
        self.problem_memory.append_event(event_payload)

        # 2. 保存状态快照
        decision_value = (
            last_decision.value
            if last_decision is not None and hasattr(last_decision, "value")
            else str(last_decision) if last_decision is not None else None
        )
        snapshot = ProblemSnapshot(
            problem_id=state.problem_id,
            iteration_count=state.iteration_count,
            status=state.status.value if state.status else "RUNNING",
            last_decision=decision_value,
        )
        self.problem_memory.save_state(snapshot)

        # 3. 输出最终 Markdown 工件
        self._save_artifact(state)

        # 4. 保存运行清单
        self._save_manifest(state, references, warning_summary)

    def _save_artifact(self, state: ProofState) -> None:
        """将 final_output 写为 Markdown 文件。"""
        from src.utils.logging.logger import save_final_output_markdown

        if not (state.final_output or "").strip():
            return
        try:
            save_final_output_markdown(
                problem_id=state.problem_id,
                final_output=state.final_output,
                runs_root=self.runs_root,
            )
        except OSError as exc:
            _logger.error("Failed to save final_output artifact: %s", exc)
 
    def _save_manifest(
        self,
        state: ProofState,
        references: list[str],
        warning_summary: str | None,
    ) -> None:
        """将终态元数据写入 manifest.json。"""

        if self.problem_memory is None:
            return
        try:
            payload = {
                "problem_id": state.problem_id,
                "iteration_count": state.iteration_count,
                "status": state.status.value if state.status else None,
                "failure_reason": state.failure_reason,
                "final_output_path": (
                    "artifact/final_output.md"
                    if (state.final_output or "").strip()
                    else None
                ),
                "references": references or [],
                "citation_warning_summary": warning_summary,
            }
            self.problem_memory.save_manifest(payload)
        except OSError as exc:
            _logger.error("Failed to save manifest artifact: %s", exc)

    # ------------------------------------------------------------------
    # 公共终态入口
    # ------------------------------------------------------------------
    def finalize_success(self, state: ProofState, *, turn_id: int) -> ProofState:
        """处理 SUCCESS：解答被判定为完全正确。"""
        state.status = RunStatus.SUCCESS
        state.failure_reason = None
        converted_solution, references, warning_summary, state.final_output = (
            self._compose(state, success=True, failure_reason=None)
        )
        state.final_answer = converted_solution
        self._persist(
            state,
            turn_id=turn_id,
            references=references,
            warning_summary=warning_summary,
            last_decision=VerificationDecision.CORRECT,
        )
        return state

    def finalize_exhausted(
        self,
        state: ProofState,
        last_decision: VerificationDecision | None = None,
        last_verification_report: str | None = None,
    ) -> ProofState:
        """处理耗尽但未通过：根据新增引理判定 PROGRESS 或 FAILED。"""
        new_lemma_count = (
            self.problem_memory.count_new_lemmas_since_start()
            if self.problem_memory
            else 0
        )
        has_progress = new_lemma_count > 0

        state.status = RunStatus.PROGRESS if has_progress else RunStatus.FAILED
        state.failure_reason = "max_turns_exhausted"

        # 统一输出生成（内部根据 progress 标志组合文本）
        converted_solution, references, warning_summary, state.final_output = (
            self._compose(
                state,
                success=False,
                failure_reason=state.failure_reason,
                progress=has_progress,
            )
        )
        state.final_answer = converted_solution if has_progress else None

        # 构建诊断事件（只保留必要信息，不再重复存储状态字面量）
        last_decision_value = (
            last_decision.value if last_decision is not None else "NONE"
        )
        extra_event: dict[str, object] = {
            "last_verifier_decision": last_decision_value,
            "last_verification_report": last_verification_report or "",
            "new_lemma_count": new_lemma_count,
        }
        if self.problem_memory is not None:
            extra_event.update(
                {
                    "initial_lemma_count": self.problem_memory.initial_lemma_count(),
                    "final_lemma_count": self.problem_memory.count_lemmas(),
                }
            )

        self._persist(
            state,
            turn_id=state.iteration_count,
            references=references,
            warning_summary=warning_summary,
            last_decision=last_decision,
            extra_event=extra_event,
        )
        return state