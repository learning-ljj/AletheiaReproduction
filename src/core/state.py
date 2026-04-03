"""核心数据结构：ProofState、VerificationDecision 与运行状态枚举。"""

from enum import Enum

from pydantic import BaseModel
from src.memory.state import ProblemSnapshot, StageSnapshot, StateValidationError


class VerificationDecision(str, Enum):
    """Verifier 的三路路由枚举。"""

    CORRECT = "CORRECT"          # 无错误 -> 终止循环
    MINOR_FLAW = "MINOR_FLAW"    # Justification Gap -> 路由至 Reviser
    CRITICAL_FLAW = "CRITICAL_FLAW"  # Critical Error -> 路由至 Generator


class RunStatus(str, Enum):
    """整题运行的终态枚举。"""

    SUCCESS = "SUCCESS"       # 完整正确解答
    PARTIAL = "PARTIAL_PROGRESS"  # 有具体进展但未完全解决（部分解答/关键引理已证明）
    FAILED = "FAILED"         # 超出能力范围 / 运行时错误


class ProofState(BaseModel):
    """整体任务状态，贯穿 Agent 生命周期。"""

    problem_id: str
    problem_text: str
    ground_truth: str | None = None
    current_proof: str = ""
    iteration_count: int = 0
    final_answer: str | None = None
    status: RunStatus | None = None
    failure_reason: str | None = None
    final_output: str | None = None


__all__ = [
    "VerificationDecision",
    "RunStatus",
    "ProofState",
    "ProblemSnapshot",
    "StageSnapshot",
    "StateValidationError",
]
