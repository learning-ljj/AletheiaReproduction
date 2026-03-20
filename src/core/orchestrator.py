"""任务编排器：只负责状态机调度，不关心具体 LLM 实现。"""

import logging
from datetime import datetime, timezone

from src.core.state import ProofState, RunStatus, VerificationLog, VerificationDecision

_logger = logging.getLogger(__name__)


class Orchestrator:
    """Aletheia 调度器。"""

    def __init__(
        self,
        max_turns: int,
        pipeline: object,
        logger: object,
        finalizer: object,
    ):
        self.max_turns = max_turns
        self.pipeline = pipeline
        self.logger = logger
        self.finalizer = finalizer

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_raw(self, problem_id: str, payload: dict) -> None:
        self.logger.append_raw_event(problem_id=problem_id, payload=payload)

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
        return state

    def _finalize_exhausted(
        self,
        state: ProofState,
        *,
        last_decision: VerificationDecision,
        last_verification_report: str,
        turn_id: int,
    ) -> ProofState:
        """轮次耗尽后调用 Final Assessor，判定 PARTIAL_PROGRESS / BEYOND_CAPABILITY。"""
        last_decision_value = last_decision.value if hasattr(last_decision, "value") else str(last_decision)

        try:
            assess_status, assess_output = self.pipeline.call_final_assessor(
                problem_text=state.problem_text,
                current_solution=state.current_proof,
                last_verifier_decision=last_decision_value,
                last_verification_report=last_verification_report,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("Final assessor failed, fallback to heuristic status: %s", exc)
            assess_status = "PARTIAL_PROGRESS" if state.current_proof else "BEYOND_CAPABILITY"
            assess_output = None

        self._append_raw(state.problem_id, {
            "agent_node": "FINAL_ASSESSOR",
            "turn_id": turn_id,
            "timestamp": self._now(),
            "last_verifier_decision": last_decision_value,
            "assessment_status": assess_status,
            "assessment_output": assess_output,
        })

        if assess_status == RunStatus.PARTIAL.value:
            state.status = RunStatus.PARTIAL
            state.failure_reason = "max_turns_exhausted"
            state.final_answer = state.current_proof
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=state.current_proof,
                failure_reason=state.failure_reason,
                partial=True,
                assessment_output=assess_output,
            )
        else:
            state.status = RunStatus.FAILED
            state.failure_reason = "beyond_capability"
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=state.current_proof,
                failure_reason=state.failure_reason,
                assessment_output=assess_output,
            )

        self._append_raw(state.problem_id, {
            "agent_node": "FINAL",
            "turn_id": turn_id,
            "timestamp": self._now(),
            "status": state.status.value,
            "failure_reason": state.failure_reason,
            "last_verifier_decision": last_decision_value,
            "final_output": state.final_output,
        })
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
            except ValueError:
                return self._finalize_failure(state, "parse_error")
            except Exception as exc:  # noqa: BLE001
                return self._finalize_failure(state, self._classify_runtime_error(exc))

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
