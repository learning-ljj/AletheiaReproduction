"""任务编排器：只负责状态机调度，不关心具体 LLM 实现。"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.core.state import ProofState, ProblemSnapshot, RunStatus, VerificationLog, VerificationDecision
from src.memory.problem_memory import ProblemMemory
from src.utils.parser import classify_parse_error, extract_xml_tag

_logger = logging.getLogger(__name__)


class Orchestrator:
    """Aletheia 调度器。"""

    def __init__(
        self,
        max_turns: int,
        pipeline: object,
        logger: object,
        finalizer: object,
        runs_root: Path | str = "runs",
    ):
        self.max_turns = max_turns
        self.pipeline = pipeline
        self.logger = logger
        self.finalizer = finalizer
        self.runs_root = Path(runs_root)
        self.problem_memory: ProblemMemory | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_raw(self, problem_id: str, payload: dict) -> None:
        if self.problem_memory is not None:
            self.problem_memory.append_event(payload)
            return
        self.logger.append_raw_event(problem_id=problem_id, payload=payload)

    def _init_problem_memory(self, problem_id: str) -> None:
        self.problem_memory = ProblemMemory(problem_id=problem_id, runs_root=self.runs_root)
        self.problem_memory.init_dirs()

    def _save_state_snapshot(
        self,
        state: ProofState,
        *,
        last_decision: VerificationDecision | str | None = None,
    ) -> None:
        if self.problem_memory is None:
            return
        decision_value = None
        if last_decision is not None:
            decision_value = last_decision.value if hasattr(last_decision, "value") else str(last_decision)
        snapshot = ProblemSnapshot(
            problem_id=state.problem_id,
            iteration_count=state.iteration_count,
            status=state.status.value if state.status is not None else "RUNNING",
            last_decision=decision_value,
        )
        self.problem_memory.save_state(snapshot)

    def _save_artifact(self, state: ProofState) -> None:
        """保存终态 final_output 到 artifact 目录。"""
        if not (state.final_output or "").strip():
            return
        try:
            self.logger.save_final_output_markdown(
                problem_id=state.problem_id,
                final_output=state.final_output,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("Failed to save final_output artifact: %s", exc)

    @staticmethod
    def _normalize_partial_solution(solution_text: str, fallback_undone: str) -> str:
        """确保 PARTIAL 的 solution 使用两段模板，并包含 <done>/<undone> 子标签。"""
        text = (solution_text or "").strip()
        done_block = extract_xml_tag(text, "done").strip()
        undone_block = extract_xml_tag(text, "undone").strip()

        if done_block and undone_block:
            return f"<done>{done_block}</done>\n<undone>{undone_block}</undone>"

        if not done_block:
            done_block = text or "无可复用的完整步骤。"
        if not undone_block:
            undone_block = (fallback_undone or "关键步骤仍缺失，无法形成完整可验证解答。").strip()

        return f"<done>{done_block}</done>\n<undone>{undone_block}</undone>"

    def _classify_runtime_error(self, exc: Exception) -> str:
        """把底层异常归一化为稳定的失败原因。"""
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, ConnectionError):
            return "llm_failure"
        msg = str(exc).lower()
        if "tool" in msg:
            return "tool_failure"
        if any(token in msg for token in ("stream", "connection", "network", "protocol")):
            return "llm_failure"
        return "parse_error"

    def _finalize_failure(self, state: ProofState, reason: str) -> ProofState:
        """统一失败收尾：设置状态并写 FINAL 事件。"""
        state.status = RunStatus.FAILED
        state.failure_reason = reason
        state.final_output = self.finalizer.build_final_output(
            success=False, solution_text=None, failure_reason=reason,
        )
        self._append_raw(state.problem_id, {
            "agent_node": "FINAL",
            "turn_id": state.iteration_count,
            "timestamp": self._now(),
            "status": state.status.value,
            "failure_reason": state.failure_reason,
            "final_output": state.final_output,
        })
        self._save_state_snapshot(state)
        self._save_artifact(state)
        return state

    def _finalize_success(self, state: ProofState, *, turn_id: int) -> ProofState:
        """统一成功收尾。"""
        state.status = RunStatus.SUCCESS
        state.failure_reason = None
        state.final_output = self.finalizer.build_final_output(
            success=True, solution_text=state.current_proof, failure_reason=None,
        )
        state.final_answer = state.current_proof
        self._append_raw(state.problem_id, {
            "agent_node": "FINAL",
            "turn_id": turn_id,
            "timestamp": self._now(),
            "status": state.status.value,
            "failure_reason": None,
            "final_output": state.final_output,
        })
        self._save_state_snapshot(state, last_decision=VerificationDecision.CORRECT)
        self._save_artifact(state)
        return state

    def _finalize_exhausted(
        self,
        state: ProofState,
        *,
        last_decision: VerificationDecision,
        last_verification_report: str,
        turn_id: int,
    ) -> ProofState:
        """轮次耗尽后仅调用一次 FINAL，判定 PARTIAL_PROGRESS / BEYOND_CAPABILITY。"""
        last_decision_value = last_decision.value if hasattr(last_decision, "value") else str(last_decision)

        try:
            final_status, final_verdict, final_solution, final_xml_output = self.pipeline.call_final(
                problem_text=state.problem_text,
                current_solution=state.current_proof,
                last_verifier_decision=last_decision_value,
                last_verification_report=last_verification_report,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("FINAL call failed, fallback to heuristic status: %s", exc)
            final_status = "PARTIAL_PROGRESS" if state.current_proof else "BEYOND_CAPABILITY"
            final_verdict = (
                "达到轮次上限，存在可复用进展。" if state.current_proof else "达到轮次上限，当前方案超出能力范围。"
            )
            fallback_undone = (
                "尚未形成可通过 verifier 的完整证明，需补全关键推理链并消除核心漏洞。"
                if state.current_proof
                else "未形成有效解答，缺乏可复用的关键引理与可验证推导。"
            )
            fallback_solution = self._normalize_partial_solution(
                (state.current_proof or "").strip(),
                fallback_undone,
            )
            final_xml_output = (
                f"<status>{final_status}</status>\n"
                f"<verdict>{final_verdict}</verdict>\n"
                f"<solution>{fallback_solution}</solution>"
            )
            final_solution = fallback_solution

        if final_status == RunStatus.PARTIAL.value:
            normalized_solution = self._normalize_partial_solution(
                final_solution,
                "尚未形成可通过 verifier 的完整证明，需补全关键推理链并消除核心漏洞。",
            )
            if normalized_solution != final_solution:
                final_solution = normalized_solution
                final_xml_output = (
                    f"<status>{final_status}</status>\n"
                    f"<verdict>{final_verdict}</verdict>\n"
                    f"<solution>{final_solution}</solution>"
                )

        if final_status == RunStatus.PARTIAL.value:
            state.status = RunStatus.PARTIAL
            state.failure_reason = "max_turns_exhausted"
            state.final_answer = final_solution
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=state.current_proof,
                failure_reason=state.failure_reason,
                partial=True,
                assessment_output=final_xml_output,
                preserve_xml=True,
            )
        else:
            state.status = RunStatus.FAILED
            state.failure_reason = "beyond_capability"
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=state.current_proof,
                failure_reason=state.failure_reason,
                assessment_output=final_xml_output,
                preserve_xml=True,
            )

        self._append_raw(state.problem_id, {
            "agent_node": "FINAL",
            "turn_id": turn_id,
            "timestamp": self._now(),
            "status": state.status.value,
            "failure_reason": state.failure_reason,
            "last_verifier_decision": last_decision_value,
            "final_status": final_status,
            "final_verdict": final_verdict,
            "xml_output": final_xml_output,
            "final_output": state.final_output,
        })
        self._save_state_snapshot(state, last_decision=last_decision)
        self._save_artifact(state)
        return state

    def _route_on_decision(self, decision: VerificationDecision, state: ProofState) -> str:
        """根据 Verifier 裁决返回下一节点名。"""
        if decision == VerificationDecision.CORRECT:
            return "FINAL"
        if decision == VerificationDecision.MINOR_FLAW:
            return "REVISER"
        if decision == VerificationDecision.CRITICAL_FLAW:
            return "GENERATOR"
        state.failure_reason = "parse_error"
        return "FINAL"

    def _record_solution_node(
        self,
        state: ProofState,
        *,
        node: str,
        turn_id: int,
        content: str | None,
        reasoning_content: str | None,
    ) -> None:
        """统一写入 GENERATOR/REVISER 的状态与 raw 事件。"""
        state.current_proof = content or ""
        state.history.append(VerificationLog(
            turn_id=turn_id, agent_node=node,
            content=content, extracted_cot=reasoning_content,
        ))
        self._append_raw(state.problem_id, {
            "agent_node": node,
            "turn_id": turn_id,
            "timestamp": self._now(),
            "content": content,
            "reasoning_content": reasoning_content,
            **({"problem_text": state.problem_text, "ground_truth": state.ground_truth}
               if node == "GENERATOR" and turn_id == 0 else {}),
        })

    def _execute_generator_node(
        self, state: ProofState, *, turn_id: int, lesson: str | None,
    ) -> None:
        """执行 Generator 节点并记录事件。"""
        resp = self.pipeline.call_generator(
            problem_text=state.problem_text,
            lesson=lesson,
        )
        self._record_solution_node(
            state, node="GENERATOR", turn_id=turn_id,
            content=resp.content, reasoning_content=resp.reasoning_content,
        )

    def _execute_reviser_node(
        self, state: ProofState, *, turn_id: int, verification_report: str,
    ) -> None:
        """执行 Reviser 节点并记录事件。"""
        resp = self.pipeline.call_reviser(
            problem_text=state.problem_text,
            previous_solution=state.current_proof,
            verification_report=verification_report,
        )
        self._record_solution_node(
            state, node="REVISER", turn_id=turn_id,
            content=resp.content, reasoning_content=resp.reasoning_content,
        )

    def _execute_verifier_node(self, state: ProofState, *, turn_id: int):
        """执行 Verifier 节点并记录事件，返回 (decision, verification_report)。"""
        verification_text, decision, verification_report, tool_trace, phase1 = \
            self.pipeline.call_verifier(
                problem_text=state.problem_text, proof_text=state.current_proof,
            )
        state.history.append(VerificationLog(
            turn_id=turn_id, agent_node="VERIFIER",
            full_verification_text=verification_text, decision=decision,
            verification_report=verification_report,
            tool_calls_trace=tool_trace, phase1_analysis=phase1,
        ))
        self._append_raw(state.problem_id, {
            "agent_node": "VERIFIER",
            "turn_id": turn_id,
            "timestamp": self._now(),
            "decision": decision.value if hasattr(decision, "value") else str(decision),
            "verification_report": verification_report,
            "tool_calls_trace": tool_trace,
            "phase1_analysis": phase1,
            "full_verification_text": verification_text,
        })
        return decision, verification_report

    def run(self, state: ProofState) -> ProofState:
        """执行调度流程：GENERATOR → VERIFIER → (REVISER|GENERATOR) → ... → FINAL。

        轮次以 Verifier 运行次数计算，最多运行 max_turns 次。
        轮次耗尽时根据最后裁决和解答内容区分结局（见 _finalize_exhausted）。
        """
        self._init_problem_memory(state.problem_id)
        self._save_state_snapshot(state)

        self._append_raw(state.problem_id, {
            "agent_node": "RUN_START",
            "turn_id": -1,
            "timestamp": self._now(),
            "problem_text": state.problem_text,
            "ground_truth": state.ground_truth,
            "max_turns": self.max_turns,
        })

        # 初始 Generator 调用（turn=0）
        try:
            self._execute_generator_node(state, turn_id=0, lesson=None)
        except Exception as exc:  # noqa: BLE001
            _logger.error("Initial generator call failed: %s", exc)
            return self._finalize_failure(state, self._classify_runtime_error(exc))

        for turn in range(1, self.max_turns + 1):
            state.iteration_count = turn
            try:
                decision, verification_report = self._execute_verifier_node(state, turn_id=turn)
            except TimeoutError:
                return self._finalize_failure(state, "timeout")
            except ValueError as exc:
                parse_error_code, failure_reason = classify_parse_error(state.current_proof, exc)
                self._append_raw(state.problem_id, {
                    "agent_node": "PARSE_ERROR",
                    "turn_id": turn,
                    "timestamp": self._now(),
                    "parse_error_code": parse_error_code,
                    "failure_reason": failure_reason,
                    "error_message": str(exc),
                })
                self._save_state_snapshot(state, last_decision=parse_error_code)

                # B22: 解析失败优先回退到 GENERATOR 做一次格式修复重试。
                if turn < self.max_turns:
                    try:
                        self._execute_generator_node(
                            state,
                            turn_id=turn,
                            lesson=(
                                "Previous verifier parsing failed with code "
                                f"{parse_error_code}. Return strict XML contract tags and valid values only."
                            ),
                        )
                        continue
                    except Exception as inner_exc:  # noqa: BLE001
                        return self._finalize_failure(state, self._classify_runtime_error(inner_exc))

                return self._finalize_failure(state, failure_reason)
            except Exception as exc:  # noqa: BLE001
                return self._finalize_failure(state, self._classify_runtime_error(exc))

            self._save_state_snapshot(state, last_decision=decision)

            next_node = self._route_on_decision(decision, state)

            if next_node == "FINAL" and decision == VerificationDecision.CORRECT:
                return self._finalize_success(state, turn_id=turn)

            if next_node == "FINAL":
                return self._finalize_failure(state, state.failure_reason or "parse_error")

            # 最后一轮：直接进入轮次耗尽处理，不再调用 GENERATOR/REVISER。
            if turn == self.max_turns:
                return self._finalize_exhausted(
                    state,
                    last_decision=decision,
                    last_verification_report=verification_report,
                    turn_id=turn,
                )

            if next_node == "REVISER":
                try:
                    self._execute_reviser_node(
                        state, turn_id=turn, verification_report=verification_report,
                    )
                except Exception as exc:  # noqa: BLE001
                    return self._finalize_failure(state, self._classify_runtime_error(exc))
                continue

            # next_node == "GENERATOR"：带着验证报告重新生成候选解。
            try:
                self._execute_generator_node(
                    state, turn_id=turn, lesson=verification_report,
                )
            except Exception as exc:  # noqa: BLE001
                _logger.error("Generator call at turn %d failed: %s", turn, exc)
                return self._finalize_failure(state, self._classify_runtime_error(exc))

        # 理论上不会到达（循环内每个出口都已 return），此处作为防御性兜底。
        return self._finalize_failure(state, "beyond_capability")
