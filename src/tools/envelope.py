"""统一工具返回包络工厂。

这个模块只做一件事：把“工具成功/失败”的返回形状固定下来，
让上层中间件和 LLM 不用再猜每个工具的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import uuid4


@dataclass(slots=True)
class ErrorEnvelope:
    """统一错误结构。

    大白话：
    - `error_code` 是机器可读的错误类型；
    - `message` 是给人看的解释；
    - `retryable` 决定中间件是否可以自动重试；
    - `detail` 放补充上下文（可选）。
    """

    error_code: str
    message: str
    retryable: bool
    detail: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def format_tool_success(*, tool: str, data: object) -> str:
    """统一成功包络。

    返回形状：
    {"status": "OK", "tool": "...", "trace_id": "...", "data": ...}
    """
    payload = {
        "status": "OK",
        "tool": tool,
        "trace_id": uuid4().hex,
        "data": data,
    }
    return json.dumps(payload, ensure_ascii=False)


def format_tool_error(
    *,
    tool: str,
    error_code: str,
    message: str,
    retryable: bool,
    detail: dict | None = None,
) -> str:
    """统一失败包络。

    返回形状：
    {
      "status": "ERROR",
      "tool": "...",
      "trace_id": "...",
      "error": {"error_code", "message", "retryable", "detail?"}
    }
    """
    error_payload = ErrorEnvelope(
        error_code=error_code,
        message=message,
        retryable=retryable,
        detail=detail,
    )
    payload = {
        "status": "ERROR",
        "tool": tool,
        "trace_id": uuid4().hex,
        "error": error_payload.to_dict(),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_tool_payload(raw: str) -> dict | None:
    """把工具返回字符串解析成字典。

    返回 None 表示不是合法 JSON 对象。
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def extract_tool_error(payload: dict | None) -> dict | None:
    """从工具包络里提取统一错误体。

    为了平滑升级，兼容两种历史形状：
    1) 新形状：{"status":"ERROR","error":{...}}
    2) 旧形状：{"status":"ERROR","error_code":...,"retryable":...}
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "ERROR":
        return None

    if isinstance(payload.get("error"), dict):
        return payload.get("error")

    if "error_code" in payload:
        return {
            "error_code": payload.get("error_code"),
            "message": payload.get("message", ""),
            "retryable": bool(payload.get("retryable")),
            "detail": payload.get("detail") if isinstance(payload.get("detail"), dict) else None,
        }
    return None


def extract_tool_success_data(payload: dict | None, *, allow_legacy: bool = False) -> object | None:
    """从工具包络里提取成功数据。

    - 新形状：返回 payload["data"]
    - allow_legacy=True 时：若遇到旧成功形状，可回退返回 payload 本身
    """
    if not isinstance(payload, dict):
        return None

    if payload.get("status") == "OK":
        return payload.get("data")

    if allow_legacy and payload.get("status") is None and payload:
        return payload

    return None
