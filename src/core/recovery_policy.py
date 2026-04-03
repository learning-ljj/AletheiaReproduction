"""Recovery and routing policies for orchestration runtime."""

from __future__ import annotations

from src.core.state import VerificationDecision


class RecoveryPolicy:
    """Centralize runtime error classification and decision routing."""

    @staticmethod
    def classify_runtime_error(exc: Exception) -> str:
        # 运行时错误分类器。
        # 大白话：把各种 Python 异常压成少量“可路由错误码”，
        # 方便 orchestrator 做统一终止策略，而不是到处写 if/else。
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

    @staticmethod
    def route_on_decision(decision: VerificationDecision) -> str:
        # 判决到节点的固定映射。
        # 大白话：
        # - CORRECT：直接收敛，进 FINAL
        # - MINOR_FLAW：局部修补，交给 REVISER
        # - CRITICAL_FLAW：结构性问题，回 GENERATOR 重做
        if decision == VerificationDecision.CORRECT:
            return "FINAL"
        if decision == VerificationDecision.MINOR_FLAW:
            return "REVISER"
        if decision == VerificationDecision.CRITICAL_FLAW:
            return "GENERATOR"
        return "FINAL"

    @staticmethod
    def build_parse_error_repair_prompt(error_message: str) -> str:
        # 解析失败时给模型的“修格式工单”。
        # 大白话：重点不是重写数学内容，而是把标签和结构修到可解析。
        return (
            "Verifier output failed XML/contract parsing. "
            "Please repair format and content expression without changing mathematically valid parts unless needed. "
            "Return valid <verdict> and <solution> blocks. "
            f"Parser detail: {error_message}"
        )
