"""任务编排器：负责装配节点执行器与调度组件。"""

import inspect
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.core.context_builder import ContextBuilder
from src.core.finalization_service import FinalizationService
from src.core.recovery_policy import RecoveryPolicy
from src.core.state import ProofState, ProblemSnapshot, VerificationDecision
from src.core.turn_loop_coordinator import TurnLoopCoordinator
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.utils.parsing.parser import (
    classify_parse_error,
    extract_xml_tag,
    parse_citation_review,
    parse_verified_lemmas,
)
from src.utils.parsing.reference_builder import build_references, export_references_bibtex

_logger = logging.getLogger(__name__)


class Orchestrator:
    """Aletheia 调度器门面。"""

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
        self.context_builder: ContextBuilder | None = None
        self.warning_messages: list[str] = []
        self.recovery_policy = RecoveryPolicy()

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
        self.context_builder = ContextBuilder(self.problem_memory)

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

    def _build_warning_summary(self, extra_warnings: list[str] | None = None) -> str | None:
        warnings = list(self.warning_messages)
        if extra_warnings:
            warnings.extend(extra_warnings)
        if not warnings:
            return None
        return "\n".join(f"- {msg}" for msg in warnings)

    @staticmethod
    def _supports_parameter(func: object, name: str) -> bool:
        try:
            return name in inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_lemma_artifact(lemma_text: str) -> str:
        text = (lemma_text or "").strip()
        if not text:
            return ""

        lowered = text.lower()
        if text.lstrip().startswith("---") and "## layer2" in lowered and "## layer3" in lowered:
            return text

        summary = " ".join(text.split())
        if len(summary) > 240:
            summary = summary[:237] + "..."

        source_match = re.search(r"(?im)^source\s*:\s*(.+)$", text)
        source = source_match.group(1).strip() if source_match else "verifier"

        lines = [
            "---",
            "title: Verified Lemma",
            f"summary: {summary}",
            "---",
            "",
            "## Layer2-Extracted",
            text,
            "",
            "## Layer3-Source",
            f"source: {source}",
            "reference: generated_from_verified_lemmas",
        ]
        return "\n".join(lines)

    def _build_references_from_solution(self, solution_text: str | None) -> tuple[str | None, list[str], list[str]]:
        if self.problem_memory is None or not (solution_text or "").strip():
            return solution_text, [], []
        try:
            converted, references, missing = build_references(solution_text or "", self.problem_memory)
            if references:
                try:
                    export_references_bibtex(references, self.problem_memory)
                except Exception as exc:  # noqa: BLE001
                    missing.append(f"bibtex_export_error: {type(exc).__name__}: {exc}")
            return converted, references, missing
        except Exception as exc:  # noqa: BLE001
            return solution_text, [], [f"reference_builder_error: {type(exc).__name__}: {exc}"]

    def _save_artifact(self, state: ProofState) -> None:
        if not (state.final_output or "").strip():
            return
        try:
            self.logger.save_final_output_markdown(
                problem_id=state.problem_id,
                final_output=state.final_output,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("Failed to save final_output artifact: %s", exc)

    def _save_manifest(
        self,
        state: ProofState,
        *,
        references: list[str] | None = None,
        warning_summary: str | None = None,
    ) -> None:
        if self.problem_memory is None:
            return
        try:
            payload = {
                "problem_id": state.problem_id,
                "iteration_count": state.iteration_count,
                "status": state.status.value if state.status else None,
                "failure_reason": state.failure_reason,
                "final_output_path": "artifact/final_output.md" if (state.final_output or "").strip() else None,
                "references": references or [],
                "citation_warning_summary": warning_summary,
            }
            self.problem_memory.save_manifest(payload)
        except Exception as exc:  # noqa: BLE001
            _logger.error("Failed to save manifest artifact: %s", exc)

    @staticmethod
    def _is_trivial_undone(text: str) -> bool:
        value = (text or "").strip().lower().strip("。.!? ")
        if not value:
            return True
        trivial_values = {
            "none",
            "n/a",
            "na",
            "no",
            "nothing",
            "无",
            "没有",
            "暂无",
            "无可补充",
        }
        return value in trivial_values

    @staticmethod
    def _normalize_partial_solution(solution_text: str, fallback_undone: str) -> str:
        text = (solution_text or "").strip()
        done_block = extract_xml_tag(text, "done").strip()
        undone_block = extract_xml_tag(text, "undone").strip()

        if done_block and undone_block and not Orchestrator._is_trivial_undone(undone_block):
            return f"<done>{done_block}</done>\n<undone>{undone_block}</undone>"

        if not done_block:
            done_block = text or "无可复用的完整步骤。"
        if not undone_block or Orchestrator._is_trivial_undone(undone_block):
            undone_block = (fallback_undone or "关键步骤仍缺失，无法形成完整可验证解答。").strip()

        return f"<done>{done_block}</done>\n<undone>{undone_block}</undone>"

    def _record_solution_node(
        self,
        state: ProofState,
        *,
        node: str,
        turn_id: int,
        content: str | None,
        reasoning_content: str | None,
        tool_calls_trace: list[dict] | None = None,
    ) -> None:
        state.current_proof = content or ""
        event_payload = {
            "agent_node": node,
            "turn_id": turn_id,
            "timestamp": self._now(),
            "content": content,
            "output_text": content,
            "tool_calls_trace": tool_calls_trace or [],
            **(
                {"problem_text": state.problem_text, "ground_truth": state.ground_truth}
                if node == "GENERATOR" and turn_id == 0
                else {}
            ),
        }

        if node != "REVISER":
            event_payload["reasoning_content"] = reasoning_content

        self._append_raw(state.problem_id, event_payload)

    def _execute_generator_node(
        self,
        state: ProofState,
        *,
        turn_id: int,
        lesson: str | None,
    ) -> None:
        call_kwargs = {
            "problem_text": state.problem_text,
            "lesson": lesson,
        }
        if self.problem_memory is not None and self._supports_parameter(self.pipeline.call_generator, "layer1_summaries"):
            if self.context_builder is not None:
                call_kwargs["layer1_summaries"] = self.context_builder.build_generator_layer1_context(summary_limit=12)
            else:
                call_kwargs["layer1_summaries"] = self.problem_memory.list_layer1_summaries(limit=12)

        resp = self.pipeline.call_generator(**call_kwargs)
        content = resp.content if hasattr(resp, "content") else str(resp)
        reasoning_content = getattr(resp, "reasoning_content", "")
        tool_calls_trace = getattr(resp, "tool_calls_trace", [])
        self._record_solution_node(
            state,
            node="GENERATOR",
            turn_id=turn_id,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls_trace=tool_calls_trace,
        )

    def _execute_reviser_node(
        self,
        state: ProofState,
        *,
        turn_id: int,
        verification_report: str,
    ) -> None:
        resp = self.pipeline.call_reviser(
            problem_text=state.problem_text,
            previous_solution=state.current_proof,
            verification_report=verification_report,
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        reasoning_content = getattr(resp, "reasoning_content", "")
        tool_calls_trace = getattr(resp, "tool_calls_trace", [])
        self._record_solution_node(
            state,
            node="REVISER",
            turn_id=turn_id,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls_trace=tool_calls_trace,
        )

    def _execute_verifier_node(self, state: ProofState, *, turn_id: int) -> tuple[VerificationDecision, str]:
        verification_text, decision, verification_report, tool_trace, phase1 = self.pipeline.call_verifier(
            problem_text=state.problem_text,
            proof_text=state.current_proof,
        )

        verified_lemmas = [
            item for item in parse_verified_lemmas(verification_text)
            if item and item.strip().upper() != "NONE"
        ]
        citation_review = parse_citation_review(verification_text).strip()

        citation_fail_count = 0
        if citation_review and citation_review.upper() != "NONE":
            try:
                citation_payload = json.loads(citation_review)
                citation_fail_count = int(citation_payload.get("fail_count", 0) or 0)
            except Exception:
                citation_fail_count = 0

        if citation_fail_count > 0:
            warning = f"Citation review reported {citation_fail_count} failed item(s) at turn {turn_id}."
            self.warning_messages.append(warning)
            self._append_raw(
                state.problem_id,
                {
                    "agent_node": "WARNING",
                    "turn_id": turn_id,
                    "timestamp": self._now(),
                    "warning_type": "citation_review",
                    "warning": warning,
                    "fail_count": citation_fail_count,
                },
            )

        if self.problem_memory is not None:
            for lemma in verified_lemmas:
                normalized_lemma = self._normalize_lemma_artifact(lemma)
                if normalized_lemma:
                    self.problem_memory.add_lemma(normalized_lemma)

        self._append_raw(
            state.problem_id,
            {
                "agent_node": "VERIFIER",
                "turn_id": turn_id,
                "timestamp": self._now(),
                "decision": decision.value if hasattr(decision, "value") else str(decision),
                "verification_report": verification_report,
                "tool_calls_trace": tool_trace,
                "phase1_analysis": phase1,
                "full_verification_text": verification_text,
                "verified_lemmas": verified_lemmas,
                "citation_review": citation_review,
            },
        )
        return decision, verification_report

    def _handle_verifier_parse_error(
        self,
        state: ProofState,
        *,
        turn_id: int,
        exc: ValueError,
    ) -> tuple[bool, str | None]:
        parse_error_code, failure_reason = classify_parse_error(state.current_proof, exc)
        self._append_raw(
            state.problem_id,
            {
                "agent_node": "PARSE_ERROR",
                "turn_id": turn_id,
                "timestamp": self._now(),
                "parse_error_code": parse_error_code,
                "failure_reason": failure_reason,
                "error_message": str(exc),
            },
        )
        self._save_state_snapshot(state, last_decision=parse_error_code)

        if turn_id >= self.max_turns:
            return False, failure_reason

        try:
            self._execute_reviser_node(
                state,
                turn_id=turn_id,
                verification_report=self.recovery_policy.build_parse_error_repair_prompt(str(exc)),
            )
            return True, None
        except Exception as reviser_exc:  # noqa: BLE001
            _logger.warning("Parse-error reviser recovery failed at turn %d: %s", turn_id, reviser_exc)
            return False, self.recovery_policy.classify_runtime_error(reviser_exc)

    def _build_finalization_service(self) -> FinalizationService:
        return FinalizationService(
            pipeline=self.pipeline,
            finalizer=self.finalizer,
            now=self._now,
            append_raw=self._append_raw,
            save_state_snapshot=self._save_state_snapshot,
            save_artifact=self._save_artifact,
            save_manifest=self._save_manifest,
            build_warning_summary=self._build_warning_summary,
            build_references_from_solution=self._build_references_from_solution,
            normalize_partial_solution=self._normalize_partial_solution,
        )

    def _build_turn_loop_coordinator(self) -> TurnLoopCoordinator:
        finalization_service = self._build_finalization_service()
        return TurnLoopCoordinator(
            max_turns=self.max_turns,
            recovery_policy=self.recovery_policy,
            execute_generator_node=self._execute_generator_node,
            execute_reviser_node=self._execute_reviser_node,
            execute_verifier_node=self._execute_verifier_node,
            handle_verifier_parse_error=self._handle_verifier_parse_error,
            save_state_snapshot=self._save_state_snapshot,
            finalize_success=finalization_service.finalize_success,
            finalize_failure=finalization_service.finalize_failure,
            finalize_exhausted=finalization_service.finalize_exhausted,
        )

    def run(self, state: ProofState) -> ProofState:
        self._init_problem_memory(state.problem_id)
        set_current_problem_memory(self.problem_memory)
        self.warning_messages = []
        self._save_state_snapshot(state)

        self._append_raw(
            state.problem_id,
            {
                "agent_node": "RUN_START",
                "turn_id": -1,
                "timestamp": self._now(),
                "problem_text": state.problem_text,
                "ground_truth": state.ground_truth,
                "max_turns": self.max_turns,
            },
        )

        coordinator = self._build_turn_loop_coordinator()
        return coordinator.run(state)
