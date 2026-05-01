"""Typed state snapshots for per-problem persistence.

This module is intentionally small for MVP.
It provides strict dictionary <-> object conversion and
structured validation errors that are easy to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from pydantic import BaseModel
# from src.memory.state import ProblemSnapshot, StageSnapshot, StateValidationError


@dataclass(slots=True)
class StateValidationError(ValueError):
    """Structured validation error for state snapshot parsing."""

    code: str
    message: str
    field: str | None = None
    detail: Any = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "detail": self.detail,
        }


def _assert_dict(data: Any, *, model_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise StateValidationError(
            code="invalid_type",
            message=f"{model_name} expects a dict input.",
            field=model_name,
            detail={"actual_type": type(data).__name__},
        )
    return data


def _assert_allowed_keys(data: dict[str, Any], allowed_keys: set[str], *, model_name: str) -> None:
    unknown = sorted(set(data.keys()) - allowed_keys)
    if unknown:
        raise StateValidationError(
            code="unknown_field",
            message=f"{model_name} contains unknown fields: {unknown}",
            field=model_name,
            detail={"unknown_fields": unknown},
        )


def _read_required_str(data: dict[str, Any], *, key: str, model_name: str) -> str:
    if key not in data:
        raise StateValidationError(
            code="missing_field",
            message=f"{model_name}.{key} is required.",
            field=key,
        )
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise StateValidationError(
            code="invalid_field_type",
            message=f"{model_name}.{key} must be a non-empty string.",
            field=key,
            detail={"actual_type": type(value).__name__},
        )
    return value


def _read_optional_str(data: dict[str, Any], *, key: str, model_name: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise StateValidationError(
            code="invalid_field_type",
            message=f"{model_name}.{key} must be a string or null.",
            field=key,
            detail={"actual_type": type(value).__name__},
        )
    return value


def _read_required_int(data: dict[str, Any], *, key: str, model_name: str) -> int:
    if key not in data:
        raise StateValidationError(
            code="missing_field",
            message=f"{model_name}.{key} is required.",
            field=key,
        )
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise StateValidationError(
            code="invalid_field_type",
            message=f"{model_name}.{key} must be an integer.",
            field=key,
            detail={"actual_type": type(value).__name__},
        )
    return value


def _read_optional_list(data: dict[str, Any], *, key: str, model_name: str) -> list[Any]:
    """读取可选列表字段，默认返回空列表。"""
    if key not in data or data[key] is None:
        return []
    value = data[key]
    if not isinstance(value, list):
        raise StateValidationError(
            code="invalid_field_type",
            message=f"{model_name}.{key} must be a list or null.",
            field=key,
            detail={"actual_type": type(value).__name__},
        )
    return value


@dataclass(slots=True)
class EventSnapshot:
    """单个阶段内的事件快照：记录执行过程中的关键信息。"""

    event_type: str  # 例如 "EXECUTION", "ERROR", "TOOL_CALL"
    status: str  # 例如 "SUCCESS", "FAILED", "PARTIAL"
    timestamp: str | None = None
    error: str | None = None  # 错误信息（若有）
    detail: dict[str, Any] | None = None  # 附加细节（工具调用、参数等）

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于JSON序列化）。"""
        return {
            "event_type": self.event_type,
            "status": self.status,
            "timestamp": self.timestamp,
            "error": self.error,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EventSnapshot":
        """从字典构造（用于JSON反序列化）。"""
        data = _assert_dict(data, model_name="EventSnapshot")
        _assert_allowed_keys(
            data,
            {"event_type", "status", "timestamp", "error", "detail"},
            model_name="EventSnapshot",
        )
        return cls(
            event_type=_read_required_str(data, key="event_type", model_name="EventSnapshot"),
            status=_read_required_str(data, key="status", model_name="EventSnapshot"),
            timestamp=_read_optional_str(data, key="timestamp", model_name="EventSnapshot"),
            error=_read_optional_str(data, key="error", model_name="EventSnapshot"),
            detail=data.get("detail"),  # 允许任意dict
        )


@dataclass(slots=True)
class StageSnapshot:
    """单个阶段的执行快照，包含该阶段的所有事件记录。"""

    stage_name: str  # 例如 "GENERATOR", "VERIFIER", "REVISER"
    turn_id: int  # 所在的迭代轮次
    status: str  # 例如 "SUCCESS", "FAILED", "PARTIAL"
    detail: str | None = None  # 阶段结果的简短摘要
    last_error: str | None = None  # 最后一个错误（用于向后兼容）
    events: list[EventSnapshot] = field(default_factory=list)  # 该阶段的所有事件列表

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "turn_id": self.turn_id,
            "status": self.status,
            "detail": self.detail,
            "last_error": self.last_error,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "StageSnapshot":
        data = _assert_dict(data, model_name="StageSnapshot")
        _assert_allowed_keys(
            data,
            {"stage_name", "turn_id", "status", "detail", "last_error", "events"},
            model_name="StageSnapshot",
        )
        events_raw = _read_optional_list(data, key="events", model_name="StageSnapshot")
        events = [EventSnapshot.from_dict(item) for item in events_raw]
        
        return cls(
            stage_name=_read_required_str(data, key="stage_name", model_name="StageSnapshot"),
            turn_id=_read_required_int(data, key="turn_id", model_name="StageSnapshot"),
            status=_read_required_str(data, key="status", model_name="StageSnapshot"),
            detail=_read_optional_str(data, key="detail", model_name="StageSnapshot"),
            last_error=_read_optional_str(data, key="last_error", model_name="StageSnapshot"),
            events=events,
        )


@dataclass(slots=True)
class ProblemSnapshot:
    """Problem-level durable state snapshot.

    Required fields are aligned with A11 task requirements:
    problem_id, iteration_count, status, last_decision.
    
    新增字段:
    - error_summary: 运行过程中所有错误的简明汇总（1-2句话）
    - stages: 每个阶段的执行快照，包含详细的事件信息
    """

    problem_id: str
    iteration_count: int
    status: str
    last_decision: str | None = None
    error_summary: str | None = None  # 错误汇总：用户友好的简短说明
    stages: list[StageSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "iteration_count": self.iteration_count,
            "status": self.status,
            "last_decision": self.last_decision,
            "error_summary": self.error_summary,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProblemSnapshot":
        data = _assert_dict(data, model_name="ProblemSnapshot")
        _assert_allowed_keys(
            data,
            {"problem_id", "iteration_count", "status", "last_decision", "error_summary", "stages"},
            model_name="ProblemSnapshot",
        )

        problem_id = _read_required_str(data, key="problem_id", model_name="ProblemSnapshot")
        status = _read_required_str(data, key="status", model_name="ProblemSnapshot")
        last_decision = _read_optional_str(data, key="last_decision", model_name="ProblemSnapshot")
        error_summary = _read_optional_str(data, key="error_summary", model_name="ProblemSnapshot")

        if "iteration_count" not in data:
            raise StateValidationError(
                code="missing_field",
                message="ProblemSnapshot.iteration_count is required.",
                field="iteration_count",
            )
        iteration_count = data["iteration_count"]
        if not isinstance(iteration_count, int) or isinstance(iteration_count, bool):
            raise StateValidationError(
                code="invalid_field_type",
                message="ProblemSnapshot.iteration_count must be an integer.",
                field="iteration_count",
                detail={"actual_type": type(iteration_count).__name__},
            )
        if iteration_count < 0:
            raise StateValidationError(
                code="invalid_field_value",
                message="ProblemSnapshot.iteration_count must be >= 0.",
                field="iteration_count",
                detail={"value": iteration_count},
            )

        stages_raw = data.get("stages", [])
        if not isinstance(stages_raw, list):
            raise StateValidationError(
                code="invalid_field_type",
                message="ProblemSnapshot.stages must be a list.",
                field="stages",
                detail={"actual_type": type(stages_raw).__name__},
            )
        stages = [StageSnapshot.from_dict(item) for item in stages_raw]

        return cls(
            problem_id=problem_id,
            iteration_count=iteration_count,
            status=status,
            last_decision=last_decision,
            error_summary=error_summary,
            stages=stages,
        )

"""核心数据结构：ProofState、VerificationDecision 与运行状态定义。"""

class VerificationDecision(str, Enum):
    """Verifier 的三路路由枚举。"""

    CORRECT = "CORRECT"          # 无错误 -> 终止循环
    MINOR_FLAW = "MINOR_FLAW"    # Justification Gap -> 路由至 Reviser
    CRITICAL_FLAW = "CRITICAL_FLAW"  # Critical Error -> 路由至 Generator


class RunStatus(str, Enum):
    """整题运行状态枚举。"""

    SUCCESS = "SUCCESS"
    PROGRESS = "PROGRESS"
    FAILED = "FAILED"


class ProofState(BaseModel):
    """整体任务状态，贯穿 Agent 生命周期。"""

    problem_id: str
    problem_text: str
    ground_truth: str | None = None
    current_proof: str = ""
    iteration_count: int = 0
    # 纯答案字段，供评测/打分脚本读取，不包含引用与告警段落。
    final_answer: str | None = None
    status: RunStatus | None = None
    failure_reason: str | None = None
    # 面向用户展示的最终文本，可能附带 References/Citation Warnings。
    final_output: str | None = None


def collect_and_generate_error_summary(state_snapshot: ProblemSnapshot) -> str | None:
    """从ProblemSnapshot中收集所有错误信息，生成用户友好的错误汇总。
    
    参数:
    - state_snapshot: 从state.json解析得到的ProblemSnapshot对象
    
    返回: 
    - 错误汇总字符串（若无错误返回None）
    
    流程：
    1. 遍历所有stages中的events，收集所有error
    2. 生成简洁的1-2句话汇总
    3. 更新state_snapshot的error_summary字段
    """
    # 收集所有错误
    all_errors: list[tuple[str, str]] = []  # (stage_name, error_msg)
    error_count_by_stage = {}
    
    for stage in state_snapshot.stages:
        for event in stage.events:
            if event.error:
                all_errors.append((stage.stage_name, event.error))
                error_count_by_stage[stage.stage_name] = error_count_by_stage.get(stage.stage_name, 0) + 1
    
    if not all_errors:
        return None
    
    # 生成错误汇总（1-2句话）
    stage_summary = ", ".join(f"{stage}({count})" for stage, count in sorted(error_count_by_stage.items()))
    error_summary = f"运行期间在以下阶段发生了{len(all_errors)}个错误: {stage_summary}。"
    
    # 如果错误过多，只展示最后3个
    if len(all_errors) > 3:
        error_details = "最后的错误包括: " + "; ".join(f"[{stage}] {msg[:60]}" for stage, msg in all_errors[-3:])
        error_summary += f" {error_details}"
    else:
        error_details = "; ".join(f"[{stage}] {msg[:60]}" for stage, msg in all_errors)
        error_summary += f" {error_details}"
    
    # 更新error_summary字段
    state_snapshot.error_summary = error_summary
    
    return error_summary

