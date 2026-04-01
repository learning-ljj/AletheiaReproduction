"""Artifact layered reader for runs/{problem_id}/artifact files."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _error(code: str, message: str, **detail) -> str:
    payload = {"error": code, "message": message}
    if detail:
        payload["detail"] = detail
    return json.dumps(payload, ensure_ascii=False)


def _is_allowed_artifact_path(path: Path) -> bool:
    # Allow only paths under runs/{problem_id}/artifact/**
    normalized = path.as_posix().lower()
    return re.search(r"(^|/)runs/[^/]+/artifact/.+", normalized) is not None


def _extract_layer1(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return ""
    first = stripped.find("\n")
    second = stripped.find("\n---", first + 1)
    if first == -1 or second == -1:
        return ""
    return stripped[: second + 4].strip()


def _extract_layer_block(text: str, layer: int) -> str:
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
    if layer not in (1, 2, 3):
        return _error("INVALID_LAYER", "layer must be one of {1,2,3}", layer=layer)

    try:
        target = Path(path).resolve(strict=True)
    except FileNotFoundError:
        return _error("PATH_NOT_FOUND", "artifact path does not exist", path=path)
    except OSError as exc:
        return _error("PATH_INVALID", f"invalid artifact path: {exc}", path=path)

    if not target.is_file():
        return _error("PATH_NOT_FILE", "artifact path must point to a file", path=str(target))

    if not _is_allowed_artifact_path(target):
        return _error(
            "PATH_NOT_ALLOWED",
            "path must stay inside runs/{problem_id}/artifact",
            path=str(target),
        )

    text = target.read_text(encoding="utf-8")
    if layer == 1:
        out = _extract_layer1(text)
    else:
        out = _extract_layer_block(text, layer=layer)

    if not out:
        return _error("LAYER_NOT_FOUND", "requested layer content is missing", layer=layer, path=str(target))
    return out
