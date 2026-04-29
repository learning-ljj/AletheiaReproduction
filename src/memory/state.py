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


@dataclass(slots=True)
class StageSnapshot:
    """Minimal per-stage snapshot for observability and routing hints."""

    stage_name: str
    status: str
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "StageSnapshot":
        data = _assert_dict(data, model_name="StageSnapshot")
        _assert_allowed_keys(
            data,
            {"stage_name", "status", "last_error"},
            model_name="StageSnapshot",
        )
        return cls(
            stage_name=_read_required_str(data, key="stage_name", model_name="StageSnapshot"),
            status=_read_required_str(data, key="status", model_name="StageSnapshot"),
            last_error=_read_optional_str(data, key="last_error", model_name="StageSnapshot"),
        )


@dataclass(slots=True)
class ProblemSnapshot:
    """Problem-level durable state snapshot.

    Required fields are aligned with A11 task requirements:
    problem_id, iteration_count, status, last_decision.
    """

    problem_id: str
    iteration_count: int
    status: str
    last_decision: str | None = None
    stages: list[StageSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "iteration_count": self.iteration_count,
            "status": self.status,
            "last_decision": self.last_decision,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProblemSnapshot":
        data = _assert_dict(data, model_name="ProblemSnapshot")
        _assert_allowed_keys(
            data,
            {"problem_id", "iteration_count", "status", "last_decision", "stages"},
            model_name="ProblemSnapshot",
        )

        problem_id = _read_required_str(data, key="problem_id", model_name="ProblemSnapshot")
        status = _read_required_str(data, key="status", model_name="ProblemSnapshot")
        last_decision = _read_optional_str(data, key="last_decision", model_name="ProblemSnapshot")

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

