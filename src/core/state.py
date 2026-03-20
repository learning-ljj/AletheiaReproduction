"""核心数据结构：ProofState, VerificationLog, VerificationDecision。"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


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


class VerificationLog(BaseModel):
    """单轮验证的完整记录，用于持久化和状态追踪。"""

    turn_id: int
    agent_node: str                              # 'GENERATOR' / 'VERIFIER' / 'REVISER'
    full_verification_text: str | None = None    # Verifier 完整输出
    decision: VerificationDecision | None = None # Verifier 裁决
    verification_report: str | None = None       # 提取的验证报告
    phase1_analysis: str | None = None           # Verifier Phase 1 初步分析文本
    tool_calls_trace: list[dict] = Field(default_factory=list)
    extracted_cot: str | None = None             # Generator/Reviser 思维链
    content: str | None = None                   # Generator/Reviser 最终解答
    parse_error: str | None = None               # 解析失败信息（严格 XML 模式）
    schema_version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProofState(BaseModel):
    """整体任务状态，贯穿 Agent 生命周期。"""

    problem_id: str
    problem_text: str
    ground_truth: str | None = None
    current_proof: str = ""
    history: list[VerificationLog] = Field(default_factory=list)
    iteration_count: int = 0
    final_answer: str | None = None
    status: RunStatus | None = None
    failure_reason: str | None = None
    final_output: str | None = None
