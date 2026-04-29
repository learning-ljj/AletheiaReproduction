"""Artifact layered reader for runs/{problem_id}/artifact files."""

from __future__ import annotations

import re
from pathlib import Path

from src.tools.envelope import format_tool_error, format_tool_success


def _is_allowed_artifact_path(path: Path) -> bool:
    # 仅允许读取 runs/{problem_id}/artifact/**，防止越权读任意磁盘文件。
    normalized = path.as_posix().lower()
    return re.search(r"(^|/)runs/[^/]+/artifact/.+", normalized) is not None


def _extract_layer1(text: str) -> str:
    # Layer1 约定为文档最前面的 YAML frontmatter：
    # ---
    # key: value
    # ---
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return ""
    first = stripped.find("\n")
    second = stripped.find("\n---", first + 1)
    if first == -1 or second == -1:
        return ""
    return stripped[: second + 4].strip()


def _extract_layer_block(text: str, layer: int) -> str:
    # Layer2/Layer3 使用 markdown 二级标题切块提取。
    if layer == 2:
        pattern = re.compile(r"(?ms)^##\s*Layer2[^\n]*\n(.*?)(?=^##\s*Layer3|\Z)")
    else:
        pattern = re.compile(r"(?ms)^##\s*Layer3[^\n]*\n(.*)$")
    match = pattern.search(text)
    return (match.group(1) if match else "").strip()


def read_artifact_layer(path: str, layer: int) -> str:
    """Read only one requested layer from an artifact markdown file.

    layer=1: YAML frontmatter summary
    layer=2: extracted theorem/proof body
    layer=3: source metadata/provenance section
    """
    # 统一策略：始终返回 envelope JSON 字符串（OK/ERROR），
    # 避免上层再维护“裸文本 / 字典 / 异常”三套分支。
    if layer not in (1, 2, 3):
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="INVALID_LAYER",
            message="layer must be one of {1,2,3}",
            retryable=False,
            detail={"layer": layer},
        )

    try:
        target = Path(path).resolve(strict=True)
    except FileNotFoundError:
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="PATH_NOT_FOUND",
            message="artifact path does not exist",
            retryable=False,
            detail={"path": path},
        )
    except OSError as exc:
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="PATH_INVALID",
            message=f"invalid artifact path: {exc}",
            retryable=False,
            detail={"path": path},
        )

    if not target.is_file():
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="PATH_NOT_FILE",
            message="artifact path must point to a file",
            retryable=False,
            detail={"path": str(target)},
        )

    if not _is_allowed_artifact_path(target):
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="PATH_NOT_ALLOWED",
            message="path must stay inside runs/{problem_id}/artifact",
            retryable=False,
            detail={"path": str(target)},
        )

    text = target.read_text(encoding="utf-8")
    if layer == 1:
        out = _extract_layer1(text)
    else:
        out = _extract_layer_block(text, layer=layer)

    if not out:
        return format_tool_error(
            tool="read_artifact_layer",
            error_code="LAYER_NOT_FOUND",
            message="requested layer content is missing",
            retryable=False,
            detail={"layer": layer, "path": str(target)},
        )

    return format_tool_success(
        tool="read_artifact_layer",
        data={
            "path": str(target),
            "layer": layer,
            "content": out,
        },
    )
