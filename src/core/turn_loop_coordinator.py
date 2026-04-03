"""Turn-loop coordinator for Generator/Verifier/Reviser orchestration."""

from __future__ import annotations

from src.core.state import ProofState, VerificationDecision


class TurnLoopCoordinator:
    """Execute the iterative turn loop and delegate node execution/finalization."""

    def __init__(
        self,
        *,
        max_turns: int,
        recovery_policy,
        execute_generator_node,
        execute_reviser_node,
        execute_verifier_node,
        handle_verifier_parse_error,
        save_state_snapshot,
        finalize_success,
        finalize_failure,
        finalize_exhausted,
    ):
        self.max_turns = max_turns
        self.recovery_policy = recovery_policy
        self.execute_generator_node = execute_generator_node
        self.execute_reviser_node = execute_reviser_node
        self.execute_verifier_node = execute_verifier_node
        self.handle_verifier_parse_error = handle_verifier_parse_error
        self.save_state_snapshot = save_state_snapshot
        self.finalize_success = finalize_success
        self.finalize_failure = finalize_failure
        self.finalize_exhausted = finalize_exhausted

    def run(self, state: ProofState) -> ProofState:
        # 冷启动：先跑一次 Generator，拿到第一版候选解答。
        # 大白话：没有初稿就没法进入 Verifier 审核闭环。
        try:
            self.execute_generator_node(state, turn_id=0, lesson=None)
        except Exception as exc:  # noqa: BLE001
            return self.finalize_failure(state, self.recovery_policy.classify_runtime_error(exc))

        # 主循环：每轮都按“Verifier 判决 -> 路由到下一节点”推进。
        # turn 从 1 开始，代表第一次审核回合。
        for turn in range(1, self.max_turns + 1):
            state.iteration_count = turn
            try:
                decision, verification_report = self.execute_verifier_node(state, turn_id=turn)
            except TimeoutError:
                return self.finalize_failure(state, "timeout")
            except ValueError as exc:
                # 这里只接“格式/解析类错误”：允许进入修复分支，而不是立刻判死。
                # 大白话：如果只是标签坏了，优先尝试修格式，不要把数学内容直接判失败。
                recovered, failure_reason = self.handle_verifier_parse_error(state, turn_id=turn, exc=exc)
                if recovered:
                    continue
                return self.finalize_failure(state, failure_reason or "parse_error")
            except Exception as exc:  # noqa: BLE001
                return self.finalize_failure(state, self.recovery_policy.classify_runtime_error(exc))

            # 每轮判决都落状态快照，确保中途中断也能回放到最近决策点。
            self.save_state_snapshot(state, last_decision=decision)

            # 决策路由：CORRECT->FINAL, MINOR_FLAW->REVISER, CRITICAL_FLAW->GENERATOR。
            next_node = self.recovery_policy.route_on_decision(decision)

            if next_node == "FINAL" and decision == VerificationDecision.CORRECT:
                return self.finalize_success(state, turn_id=turn)

            if next_node == "FINAL":
                # 非 CORRECT 但被路由到 FINAL，说明属于不可恢复或策略性终止。
                state.failure_reason = state.failure_reason or "parse_error"
                return self.finalize_failure(state, state.failure_reason)

            if turn == self.max_turns:
                # 到达预算上限：不再继续修订，走 exhausted 终态。
                return self.finalize_exhausted(
                    state,
                    last_decision=decision,
                    last_verification_report=verification_report,
                    turn_id=turn,
                )

            if next_node == "REVISER":
                # 轻缺陷走 Reviser：目标是“微调修补”，不推翻整稿。
                try:
                    self.execute_reviser_node(state, turn_id=turn, verification_report=verification_report)
                except Exception as exc:  # noqa: BLE001
                    return self.finalize_failure(state, self.recovery_policy.classify_runtime_error(exc))
                continue

            # CRITICAL_FLAW 走 Generator：要求基于 verifier 报告做更大幅重生。
            try:
                self.execute_generator_node(state, turn_id=turn, lesson=verification_report)
            except Exception as exc:  # noqa: BLE001
                return self.finalize_failure(state, self.recovery_policy.classify_runtime_error(exc))

        # 理论兜底：循环意外走穿时，按 beyond_capability 终止。
        return self.finalize_failure(state, "beyond_capability")
