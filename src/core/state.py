"""核心数据结构：ProofState, VerificationLog, VerificationDecision。"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class VerificationDecision(str, Enum):
    """Verifier 的三路路由枚举。"""

    CORRECT = "CORRECT"          # 无错误 -> 终止循环
    MINOR_FLAW = "MINOR_FLAW"    # Justification Gap -> 路由至 Reviser
    CRITICAL_FLAW = "CRITICAL_FLAW"  # Critical Error -> 路由至 Generator


class VerificationLog(BaseModel):
    """单轮验证的完整记录，用于持久化和状态追踪。"""

    turn_id: int
    agent_node: str                              # 'GENERATOR' / 'VERIFIER' / 'REVISER'
    full_verification_text: str | None = None    # Verifier 完整输出
    decision: VerificationDecision | None = None # Verifier 裁决
    bug_report: str | None = None                # 提取的问题摘要
    phase1_analysis: str | None = None           # Verifier Phase 1 初步分析文本
    tool_calls_trace: list[dict] = Field(default_factory=list)
    extracted_cot: str | None = None             # Generator/Reviser 思维链
    content: str | None = None                   # Generator/Reviser 最终解答
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ProofState(BaseModel):
    """整体任务状态，贯穿 Agent 生命周期。"""

    problem_id: str
    problem_text: str
    current_proof: str = ""
    history: list[VerificationLog] = Field(default_factory=list)
    iteration_count: int = 0
    final_answer: str | None = None
