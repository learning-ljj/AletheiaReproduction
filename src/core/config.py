"""配置加载器：读取 YAML 配置文件，自动替换 ${ENV_VAR} 占位符为环境变量值。"""

import os
import re
from pathlib import Path
import yaml


def _substitute_env_vars(obj):
    """递归遍历 dict/list/str，将 ${VAR} 替换为环境变量值。"""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


def load_config(path: str = "config/settings.yaml") -> dict:
    """加载 YAML 配置，自动将 ${VAR} 替换为环境变量值。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute_env_vars(raw)


def _load_yaml_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt yaml must be a mapping object: {path}")
    return loaded


def _load_prompt_dir(prompt_dir: Path) -> dict:
    merged: dict = {}
    for file_path in sorted(prompt_dir.glob("*.yaml")):
        section = _load_yaml_file(file_path)
        for key, value in section.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def load_prompts(path: str = "config/prompts") -> dict:
    """加载 prompt 模板，仅支持目录或显式 YAML 文件。"""
    target = Path(path)

    if target.is_dir():
        prompts = _load_prompt_dir(target)
        if prompts:
            return prompts

    if target.is_file():
        return _load_yaml_file(target)

    raise FileNotFoundError(f"Prompt config not found: {path}")
