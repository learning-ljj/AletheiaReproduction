"""Finalization service for orchestration terminal states."""

from __future__ import annotations

import logging

from src.core.state import ProofState, RunStatus, VerificationDecision

_logger = logging.getLogger(__name__)


class FinalizationService:
    """Encapsulate SUCCESS/FAILED/PARTIAL finalization and artifact persistence."""

    def __init__(
        self,
        *,
        pipeline,
        finalizer,
        now,
        append_raw,
        save_state_snapshot,
        save_artifact,
        save_manifest,
        build_warning_summary,
        build_references_from_solution,
        normalize_partial_solution,
    ):
        self.pipeline = pipeline
        self.finalizer = finalizer
        self.now = now
        self.append_raw = append_raw
        self.save_state_snapshot = save_state_snapshot
        self.save_artifact = save_artifact
        self.save_manifest = save_manifest
        self.build_warning_summary = build_warning_summary
        self.build_references_from_solution = build_references_from_solution
        self.normalize_partial_solution = normalize_partial_solution

    def finalize_failure(self, state: ProofState, reason: str) -> ProofState:
        state.status = RunStatus.FAILED
        state.failure_reason = reason
        converted_solution, references, reference_warnings = self.build_references_from_solution(state.current_proof)
        warning_summary = self.build_warning_summary(reference_warnings)
        state.final_output = self.finalizer.build_final_output(
            success=False,
            solution_text=converted_solution,
            failure_reason=reason,
            references=references,
            warning_summary=warning_summary,
        )
        self.append_raw(
            state.problem_id,
            {
                "agent_node": "FINAL",
                "turn_id": state.iteration_count,
                "timestamp": self.now(),
                "status": state.status.value,
                "failure_reason": state.failure_reason,
                "final_output": state.final_output,
            },
        )
        self.save_state_snapshot(state)
        self.save_artifact(state)
        self.save_manifest(state, references=references, warning_summary=warning_summary)
        return state

    def finalize_success(self, state: ProofState, *, turn_id: int) -> ProofState:
        state.status = RunStatus.SUCCESS
        state.failure_reason = None
        converted_solution, references, reference_warnings = self.build_references_from_solution(state.current_proof)
        warning_summary = self.build_warning_summary(reference_warnings)
        state.final_output = self.finalizer.build_final_output(
            success=True,
            solution_text=converted_solution,
            failure_reason=None,
            references=references,
            warning_summary=warning_summary,
        )
        state.final_answer = converted_solution
        self.append_raw(
            state.problem_id,
            {
                "agent_node": "FINAL",
                "turn_id": turn_id,
                "timestamp": self.now(),
                "status": state.status.value,
                "failure_reason": None,
                "final_output": state.final_output,
            },
        )
        self.save_state_snapshot(state, last_decision=VerificationDecision.CORRECT)
        self.save_artifact(state)
        self.save_manifest(state, references=references, warning_summary=warning_summary)
        return state

    def finalize_exhausted(
        self,
        state: ProofState,
        *,
        last_decision: VerificationDecision,
        last_verification_report: str,
        turn_id: int,
    ) -> ProofState:
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
            fallback_solution = self.normalize_partial_solution(
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
            normalized_solution = self.normalize_partial_solution(
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

        converted_solution, references, reference_warnings = self.build_references_from_solution(state.current_proof)
        warning_summary = self.build_warning_summary(reference_warnings)

        if final_status == RunStatus.PARTIAL.value:
            state.status = RunStatus.PARTIAL
            state.failure_reason = "max_turns_exhausted"
            state.final_answer = final_solution
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=converted_solution,
                failure_reason=state.failure_reason,
                partial=True,
                assessment_output=final_xml_output,
                references=references,
                warning_summary=warning_summary,
            )
        else:
            state.status = RunStatus.FAILED
            state.failure_reason = "beyond_capability"
            state.final_output = self.finalizer.build_final_output(
                success=False,
                solution_text=converted_solution,
                failure_reason=state.failure_reason,
                assessment_output=final_xml_output,
                references=references,
                warning_summary=warning_summary,
            )

        self.append_raw(
            state.problem_id,
            {
                "agent_node": "FINAL",
                "turn_id": turn_id,
                "timestamp": self.now(),
                "status": state.status.value,
                "failure_reason": state.failure_reason,
                "last_verifier_decision": last_decision_value,
                "final_status": final_status,
                "final_verdict": final_verdict,
                "xml_output": final_xml_output,
                "final_output": state.final_output,
            },
        )
        self.save_state_snapshot(state, last_decision=last_decision)
        self.save_artifact(state)
        self.save_manifest(state, references=references, warning_summary=warning_summary)
        return state
