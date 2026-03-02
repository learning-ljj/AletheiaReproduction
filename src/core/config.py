"""配置加载器：读取 YAML 配置文件，自动替换 ${ENV_VAR} 占位符为环境变量值。"""

import os
import re
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


def load_prompts(path: str = "config/prompts.yaml") -> dict:
    """加载所有 prompt 模板，返回 dict。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
