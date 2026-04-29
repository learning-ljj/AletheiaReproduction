"""任务编排器：执行主循环并协调节点执行与终态持久化。"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.context_builder import ContextBuilder
from src.core.finalizer import FinalizerEngine
from src.core.recovery_policy import RecoveryPolicy
from src.memory.state import ProofState, ProblemSnapshot, VerificationDecision
from src.memory.problem_memory import ProblemMemory, set_current_problem_memory
from src.utils.parsing.parser import parse_citation_review,parse_verified_lemmas


class Orchestrator:
    """Aletheia 调度器门面。"""

    def __init__(
        self,
        max_turns: int,
        pipeline: object,
        runs_root: Path | str = "runs",
    ):
        """初始化 Orchestrator。

        参数:
        - max_turns: 最大的验证/修订回合数（不含初始生成回合）。
        - pipeline: 提供 `call_generator`, `call_reviser`, `call_verifier` 等方法的流水线对象。
        - runs_root: 运行产物与问题存档的根目录（路径或字符串）。

        内部逻辑:
        - 初始化 `problem_memory`、`context_builder` 占位符以及 `RecoveryPolicy`。
        """
        self.max_turns = max_turns
        self.pipeline = pipeline
        self.runs_root = Path(runs_root)
        self.problem_memory: ProblemMemory | None = None
        self.context_builder: ContextBuilder | None = None
        self.warning_messages: list[str] = []
        self.recovery_policy = RecoveryPolicy()

    def _now(self) -> str:
        """返回当前 UTC 时间的 ISO8601 字符串表示。

        返回示例: '2026-04-16T12:34:56.789012+00:00'。
        """
        return datetime.now(timezone.utc).isoformat()

    def _save_state_snapshot(
        self,
        state: ProofState,
        *,
        last_decision: VerificationDecision | str | None = None,
    ) -> None:
        """将当前 ProofState 的快照保存到 ProblemMemory。

        - 如果 `problem_memory` 未初始化，则不执行任何操作。
        - `last_decision` 可以是枚举（取 `.value`）或字符串，用于记录上一次验证决策。
        - 构造 `ProblemSnapshot` 并调用 `ProblemMemory.save_state`。
        """
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

    @staticmethod
    def _build_citation_warning_lines(citation_payload: dict, turn_id: int) -> list[str]:
        """构建对用户可见的引用告警明细行。"""
        fail_count = int(citation_payload.get("fail_count", 0) or 0)
        lines = [f"Citation review reported {fail_count} failed item(s) at turn {turn_id}."]

        items = citation_payload.get("items") if isinstance(citation_payload, dict) else []
        if not isinstance(items, list):
            return lines

        idx = 1
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("passed") is True:
                continue

            cite_path = str(item.get("cite") or item.get("path") or "unknown_path")
            reason = str(item.get("reason") or "UNKNOWN_REASON")
            detail = str(item.get("detail") or item.get("message") or "").strip()
            line = f"[{idx}] {cite_path}: {reason}"
            if detail:
                line += f" ({detail})"
            lines.append(line)
            idx += 1
        return lines

    def _execute_solution_node(
        self,
        state: ProofState,
        *,
        node: str,
        turn_id: int,
        lesson: str | None = None,
        verification_report: str | None = None,
    ) -> None:
        """执行一个解答节点并记录其输出。

        - node: 'GENERATOR' 或 'REVISER'。
        - 对于 GENERATOR，优先尝试从 `context_builder` 获取引理上下文，否则从 `problem_memory` 列表获取。
        - 对于 REVISER，同样注入引理上下文，帮助小修阶段参考既有引理与证明。
        - 调用 `pipeline.call_generator` 或 `pipeline.call_reviser` 获取 `resp`，提取 `content`、`reasoning_content` 与可选 `tool_calls_trace`。
        - 直接更新 `state.current_proof` 并写入原始事件日志。
        """
        if node == "GENERATOR":
            lemma_context_items: list[str] | None = None
            if self.context_builder is not None:
                lemma_context_items = self.context_builder.build_generator_lemma_context(item_limit=12)
            elif self.problem_memory is not None:
                lemma_context_items = self.problem_memory.list_lemma_context_items(limit=12)
            resp = self.pipeline.call_generator(
                problem_text=state.problem_text,
                lesson=lesson,
                lemma_context_items=lemma_context_items,
            )
        else:
            lemma_context_items: list[str] | None = None
            if self.context_builder is not None:
                lemma_context_items = self.context_builder.build_reviser_lemma_context(item_limit=12)
            elif self.problem_memory is not None:
                lemma_context_items = self.problem_memory.list_lemma_context_items(limit=12)
            resp = self.pipeline.call_reviser(
                problem_text=state.problem_text,
                previous_solution=state.current_proof,
                verification_report=verification_report or "",
                lemma_context_items=lemma_context_items,
            )

        content = resp.content if hasattr(resp, "content") else str(resp)
        reasoning_content = getattr(resp, "reasoning_content", "")
        tool_calls_trace = getattr(resp, "tool_calls_trace", [])
        state.current_proof = content or ""
        event_payload = {
            "node": node,
            "turn_id": turn_id,
            "timestamp": self._now(),
            "content": content,
            "tool_calls_trace": tool_calls_trace or [],
            **(
                {"problem_text": state.problem_text, "ground_truth": state.ground_truth}
                if node == "GENERATOR" and turn_id == 0
                else {}
            ),
        }
        if node != "REVISER":
            event_payload["reasoning_content"] = reasoning_content
        self.problem_memory.append_event(event_payload)

    def _execute_verifier_node(self, state: ProofState, *, turn_id: int) -> tuple[VerificationDecision, str]:
        """调用验证器并处理验证结果与产物。

        返回:
        - tuple(decision, verification_report)

        处理流程:
        - 调用 `pipeline.call_verifier` 获取原始验证文本、决策与报告。
        - 使用解析器提取 `verified_lemmas` 与 `citation_review`。
        - 若 `citation_review` 为 JSON 且包含失败计数，则生成 WARNING 事件并记录到 `warning_messages`。
        - 将规范化的 verified lemmas（如果有）写入 `ProblemMemory` 以供后续引用。
        - 把完整的验证信息写入原始事件日志并返回决策与报告。
        """
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
        citation_payload: dict = {}
        if citation_review and citation_review.upper() != "NONE":
            try:
                citation_payload = json.loads(citation_review)
                citation_fail_count = int(citation_payload.get("fail_count", 0) or 0)
            except (TypeError, ValueError):
                citation_fail_count = 0
                citation_payload = {}

        if citation_fail_count > 0:
            warning_lines = self._build_citation_warning_lines(citation_payload, turn_id)
            self.warning_messages.extend(warning_lines)
            self.problem_memory.append_event(
                {
                    "node": "WARNING",
                    "turn_id": turn_id,
                    "timestamp": self._now(),
                    "warning_type": "citation_review",
                    "warning": warning_lines[0],
                    "fail_count": citation_fail_count,
                    "warning_details": warning_lines[1:],
                },
            )

        if self.problem_memory is not None:
            for lemma in verified_lemmas:
                normalized_lemma = (lemma or "").strip()
                if normalized_lemma:
                    self.problem_memory.add_lemma(normalized_lemma)

        self.problem_memory.append_event(
            {
                "node": "VERIFIER",
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

    def run(self, state: ProofState) -> ProofState:
        """外部调用入口：为指定问题运行完整的生成—验证—修订回合并返回最终状态。

        行为:
        - 初始化 `ProblemMemory`（并设置为当前上下文），重置警告列表并保存初始状态快照。
        - 记录 `RUN_START` 事件以标记运行的开始。
        - 构造 `FinalizerEngine` 并直接执行主循环。
        - 运行异常时抛出友好错误提示；运行结束后清理当前问题内存引用。
        """
        def _finish(result: ProofState) -> ProofState:
            set_current_problem_memory(None)
            return result

        def _raise_runtime(stage_name: str, exc: Exception) -> None:
            set_current_problem_memory(None)
            raise RuntimeError(f"运行失败（{stage_name}）：{type(exc).__name__}: {exc}") from exc

        # 初始化 ProblemMemory 并构建 ContextBuilder 与 FinalizerEngine
        self.problem_memory = ProblemMemory(problem_id=state.problem_id, runs_root=self.runs_root)
        self.problem_memory.init_dirs()
        self.warning_messages = []
        self.context_builder = ContextBuilder(self.problem_memory)
        self.finalizer_engine = FinalizerEngine(
            problem_memory=self.problem_memory,
            warnings=self.warning_messages,
            runs_root=self.runs_root,
        )
        set_current_problem_memory(self.problem_memory)
        self._save_state_snapshot(state)

        self.problem_memory.append_event(
            {
                "node": "RUN_START",
                "turn_id": 0,
                "timestamp": self._now(),
                "problem_text": state.problem_text,
                "ground_truth": state.ground_truth,
                "max_turns": self.max_turns,
            },
        )

        try:
            self._execute_solution_node(state, node="GENERATOR", turn_id=0, lesson=None)
        except Exception as exc:  # noqa: BLE001
            _raise_runtime("GENERATOR@turn0", exc)

        decision = VerificationDecision.CRITICAL_FLAW  # 初始默认决策，确保未进入循环时也能完整运行
        verification_report = ""  # 同上，确保即使验证器完全失败也能进入循环并正确走 finalizer 的非成功分支。

        for turn in range(1, self.max_turns + 1):
            state.iteration_count = turn
            try:
                decision, verification_report = self._execute_verifier_node(state, turn_id=turn)
            except TimeoutError as exc:
                _raise_runtime("VERIFIER_TIMEOUT", exc)
            except Exception as exc:  # noqa: BLE001
                _raise_runtime("VERIFIER", exc)

            self._save_state_snapshot(state, last_decision=decision)
            next_node = self.recovery_policy.route_on_decision(decision)

            if next_node == "FINAL" and decision == VerificationDecision.CORRECT:
                result = self.finalizer_engine.finalize_success(state, turn_id=turn)
                return _finish(result)

            if next_node == "REVISER" and turn < self.max_turns:
                try:
                    self._execute_solution_node(
                        state,
                        node="REVISER",
                        turn_id=turn,
                        verification_report=verification_report,
                    )
                except Exception as exc:  # noqa: BLE001
                    _raise_runtime("REVISER", exc)
                continue

            if next_node == "GENERATOR" and turn < self.max_turns:
                try:
                    self._execute_solution_node(
                        state,
                        node="GENERATOR",
                        turn_id=turn,
                        lesson=verification_report,
                    )
                except Exception as exc:  # noqa: BLE001
                    _raise_runtime("GENERATOR", exc)
                continue

        # 达到最大轮次后，统一走 finalizer 的非成功收敛入口。
        result = self.finalizer_engine.finalize_exhausted(
            state,
            last_decision=decision,
            last_verification_report=verification_report,
        )
        return _finish(result)
