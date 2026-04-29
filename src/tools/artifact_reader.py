"""Artifact layered reader for runs/{problem_id}/artifact files."""

"""
lemma 文档（三层）示例：

~~~markdown
---
summary: 若 n 为正整数，则 gcd(n, n+1)=1
conditions:
  - n is positive integer
conclusion: gcd(n, n+1)=1
source: self_proved
---

## Layer2-Proof
Step 1. 设 d 同时整除 n 与 n+1，则 d 整除 (n+1)-n=1。
Step 2. 因此 d=1，故 gcd(n,n+1)=1。

## Layer3-Source
Source: generator
Reference: self_proved
~~~

paper 文档（三层）示例：

~~~markdown
---
arxiv_id: 2501.12345
summary: 在条件 A,B 下，得到界 O(n log n)
conditions:
  - assumption A
  - assumption B
conclusion: bound O(n log n)
---

## Layer2-Extracted
Theorem ...
Proof ...

## Layer3-Source
title: Sample Paper
authors: Alice; Bob
url: https://arxiv.org/abs/2501.12345
~~~
"""

from __future__ import annotations

import re
from pathlib import Path

from src.tools.format import format_tool_error, format_tool_success


def _is_allowed_artifact_path(path: Path) -> bool:
    # 仅允许读取 runs/{problem_id}/artifact/**，防止越权读任意磁盘文件。
    normalized = path.as_posix().lower()
    return re.search(r"(^|/)runs/[^/]+/artifact/.+", normalized) is not None


def _extract_yaml(text: str) -> str:
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


def _extract_layer(text: str, layer: int) -> str:
    # Layer2/Layer3 使用 markdown 二级标题切块提取。
    if layer == 2:
        pattern = re.compile(r"(?ms)^##\s*Layer2[^\n]*\n(.*?)(?=^##\s*Layer3|\Z)")
    else:
        pattern = re.compile(r"(?ms)^##\s*Layer3[^\n]*\n(.*)$")
    match = pattern.search(text)
    return (match.group(1) if match else "").strip()


def read_artifact(path: str, layer: int) -> str:
    """Read only one requested layer from an artifact markdown file.

    layer=1: YAML frontmatter summary
    layer=2: extracted theorem/proof body
    layer=3: source metadata/provenance section
    """
    # 统一策略：始终返回 format JSON 字符串（OK/ERROR），避免上层再维护“裸文本 / 字典 / 异常”三套分支。

    # --- 1. 参数校验 ---
    if layer not in (1, 2, 3):
        return format_tool_error(
            tool="read_artifact",
            error_code="INVALID_LAYER",
            message="layer must be one of {1,2,3}",
            retryable=False,
            detail={"layer": layer},
        )

    # --- 2. 路径解析与存在性检查 ---
    try:
        target = Path(path).resolve(strict=True)
    except FileNotFoundError:
        return format_tool_error(
            tool="read_artifact",
            error_code="PATH_NOT_FOUND",
            message="artifact path does not exist",
            retryable=False,
            detail={"path": path},
        )
    except OSError as exc:
        return format_tool_error(
            tool="read_artifact",
            error_code="PATH_INVALID",
            message=f"invalid artifact path: {exc}",
            retryable=False,
            detail={"path": path},
        )

    if not target.is_file():
        return format_tool_error(
            tool="read_artifact",
            error_code="PATH_NOT_FILE",
            message="artifact path must point to a file",
            retryable=False,
            detail={"path": str(target)},
        )

    # --- 3. 路径安全校验 ---
    if not _is_allowed_artifact_path(target):
        return format_tool_error(
            tool="read_artifact",
            error_code="PATH_NOT_ALLOWED",
            message="path must stay inside runs/{problem_id}/artifact",
            retryable=False,
            detail={"path": str(target)},
        )

    # --- 4. 读取文件并按层提取 ---
    text = target.read_text(encoding="utf-8")
    if layer == 1:
        out = _extract_yaml(text)
    else:
        out = _extract_layer(text, layer=layer)

    if not out:
        return format_tool_error(
            tool="read_artifact",
            error_code="LAYER_NOT_FOUND",
            message="requested layer content is missing",
            retryable=False,
            detail={"layer": layer, "path": str(target)},
        )

    # --- 5. 成功返回 ---
    return format_tool_success(
        tool="read_artifact",
        data={
            "path": str(target),
            "layer": layer,
            "content": out,
        },
    )
