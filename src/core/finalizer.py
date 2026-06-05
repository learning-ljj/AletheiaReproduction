"""Final output helpers and terminal-state finalization engine.

负责将运行终态（SUCCESS、PROGRESS、FAILED）转化为最终输出文本，并持久化状态与工件。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from src.memory.state import ProofState, RunStatus, VerificationDecision, ProblemSnapshot, StageSnapshot


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

        from src.utils.parsing.build_reference import build_references, references_to_bibtex

        try:
            converted, references, missing = build_references(
                solution_text or "", self.problem_memory
            )
            if references:
                try:
                    self.problem_memory.save_bibtex(references_to_bibtex(references))
                except Exception as exc:
                    missing.append(
                        f"bibtex_export_error: {type(exc).__name__}: {exc}"
                    )
            return converted, references, missing
        except Exception as exc:
            return (
                solution_text,
                [],
                [f"build_reference_error: {type(exc).__name__}: {exc}"],
            )

    def _build_final_output(
        self,
        *,
        success: bool,
        solution_text: str | None,
        failure_reason: str | None,
        verifier_text: str | None = None,
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
        solution_body = (solution_text or "").strip()
        verifier_body = (verifier_text or "").strip()

        sections: list[str] = []

        if not success:
            if progress and solution_body:
                sections.append(
                    f"**Status**: PROGRESS\n**Failure Reason**: {reason}"
                )
            else:
                sections.append(f"**Status**: FAILED\n**Failure Reason**: {reason}")

        if verifier_body:
            sections.append("## Last Verifier Output\n" + verifier_body)

        if solution_body:
            if verifier_body or not success:
                sections.append("## Solution\n" + solution_body)
            else:
                sections.append(solution_body)

        output = "\n\n".join(section.strip() for section in sections if section.strip())

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
        verifier_text: str | None = None,
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
            verifier_text=verifier_text,
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
        stages: list[StageSnapshot] | None = None,
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
            stages=list(stages or []),
        )
        self.problem_memory.save_state(snapshot)

        # 3. 输出最终 Markdown 工件
        self._save_artifact(state)

        # 4. 保存运行清单
        self._save_manifest(state, references, warning_summary)

    def _save_artifact(self, state: ProofState) -> None:
        """将 final_output 写为 Markdown 文件。"""
        import re

        output = (state.final_output or "").strip()
        if not output:
            return

        # 将展示数学的 LaTeX 标记从 \[...\] 转换为 $$...$$
        normalized = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", output, flags=re.S)

        artifact_dir = self.runs_root / state.problem_id / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target_path = artifact_dir / "final_output.md"
        try:
            target_path.write_text(normalized + "\n", encoding="utf-8")
        except OSError:
            pass
 
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
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 公共终态入口
    # ------------------------------------------------------------------
    def finalize_success(
        self,
        state: ProofState,
        *,
        turn_id: int,
        last_verifier_text: str | None = None,
        stages: list[StageSnapshot] | None = None,
    ) -> ProofState:
        """处理 SUCCESS：解答被判定为完全正确。"""
        state.status = RunStatus.SUCCESS
        state.failure_reason = None
        converted_solution, references, warning_summary, state.final_output = (
            self._compose(
                state,
                success=True,
                failure_reason=None,
                verifier_text=last_verifier_text,
            )
        )
        state.final_answer = converted_solution
        self._persist(
            state,
            turn_id=turn_id,
            references=references,
            warning_summary=warning_summary,
            last_decision=VerificationDecision.CORRECT,
            stages=stages,
        )
        return state

    def finalize_exhausted(
        self,
        state: ProofState,
        last_decision: VerificationDecision | None = None,
        last_verification: str | None = None,
        last_verifier_text: str | None = None,
        stages: list[StageSnapshot] | None = None,
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
                verifier_text=last_verifier_text,
            )
        )
        state.final_answer = converted_solution if has_progress else None

        # 构建诊断事件（只保留必要信息，不再重复存储状态字面量）
        last_decision_value = (
            last_decision.value if last_decision is not None else "NONE"
        )
        extra_event: dict[str, object] = {
            "last_verifier_decision": last_decision_value,
            "last_verification": last_verification or "",
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
            stages=stages,
        )
        return state