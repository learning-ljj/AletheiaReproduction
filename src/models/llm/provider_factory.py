"""Provider selection helpers and SDK client bundle construction for LLMClient."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from openai import OpenAI


@dataclass(frozen=True)
class ProviderBundle:
    """Resolved provider runtime dependencies for LLMClient."""

    client: OpenAI
    model: str
    thinking: bool
    max_tokens: int


class ProviderFactory:
    """Factory that resolves provider config and creates OpenAI SDK clients."""

    @staticmethod
    def _configured(value: str | None) -> bool:
        return bool(value) and (not str(value).startswith("${"))

    @staticmethod
    def _contains_placeholder(value: str | None) -> bool:
        return isinstance(value, str) and "${" in value

    @staticmethod
    def _resolve_provider_config(config: dict, provider_name: str) -> dict:
        # 合并顺序：shared_defaults < provider_config，厂商专属配置可以覆盖全局默认值。
        shared_defaults = config.get("llm_defaults") or {}
        provider_config = config.get(provider_name) or {}
        return {**shared_defaults, **provider_config}

    @classmethod
    def resolve_provider_payload(cls, config: dict) -> dict:
        """Resolve provider from full settings and normalize to {'provider': cfg}."""
        # 约定：配置中 provider 必须是字符串（如 deepseek/volcano）。归一化后的返回值始终使用固定键 'provider'，这样下游 LLMClient 不再关心具体厂商字段名。
        provider = config.get("provider")

        if provider == "volcano":
            volcano_cfg = cls._resolve_provider_config(config, "volcano")
            api_key = volcano_cfg.get("api_key")
            base_url = volcano_cfg.get("base_url")
            if not api_key or cls._contains_placeholder(api_key):
                raise ValueError(
                    "Volcano provider selected but `volcano.api_key` is missing or contains placeholder. "
                    "Ensure .env contains VOLCANO_API_KEY and that you called load_dotenv() before loading config."
                )
            if base_url and not str(base_url).startswith("http"):
                raise ValueError(
                    f"Volcano base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
                )
            return {"provider": volcano_cfg}

        if provider == "deepseek":
            deepseek_cfg = cls._resolve_provider_config(config, "deepseek")
            api_key = deepseek_cfg.get("api_key")
            base_url = deepseek_cfg.get("base_url")
            if not api_key or cls._contains_placeholder(api_key):
                raise ValueError(
                    "DeepSeek provider selected but `deepseek.api_key` is missing or contains placeholder. "
                    "Ensure .env contains DEEPSEEK_API_KEY and that you called load_dotenv() before loading config."
                )
            if base_url and not str(base_url).startswith("http"):
                raise ValueError(
                    f"DeepSeek base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
                )
            return {"provider": deepseek_cfg}

        if provider == "litellm":
            litellm_cfg = cls._resolve_provider_config(config, "litellm")
            api_key = litellm_cfg.get("api_key")
            base_url = litellm_cfg.get("base_url")
            model = litellm_cfg.get("model")
            if not api_key or cls._contains_placeholder(api_key):
                raise ValueError(
                    "LiteLLM provider selected but `litellm.api_key` is missing or contains placeholder. "
                    "Ensure .env contains OPENAI_API_KEY and that you called load_dotenv() before loading config."
                )
            if not base_url or cls._contains_placeholder(base_url):
                raise ValueError(
                    "LiteLLM provider selected but `litellm.base_url` is missing or contains placeholder. "
                    "Ensure .env contains OPENAI_BASE_URL."
                )
            if base_url and not str(base_url).startswith("http"):
                raise ValueError(
                    f"LiteLLM base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
                )
            if not model or cls._contains_placeholder(model):
                raise ValueError(
                    "LiteLLM provider selected but `litellm.model` is missing or contains placeholder. "
                    "Ensure .env contains LITELLM_MODEL."
                )
            return {"provider": litellm_cfg}

        raise ValueError(f"Unknown LLM provider: {provider!r}.")

    @staticmethod
    def build_client_bundle(provider_payload: dict) -> ProviderBundle:
        """Create OpenAI SDK client and stream/runtime options from normalized payload."""
        # provider_payload 形如 {"provider": {...}}。
        cfg = provider_payload["provider"]
        # 配置值统一做显式类型转换，并在缺失时回落默认值。
        connect_timeout = float(cfg.get("connect_timeout_seconds", 30.0))
        read_timeout = float(cfg.get("read_timeout_seconds", 90.0))
        write_timeout = float(cfg.get("write_timeout_seconds", read_timeout))
        pool_timeout = float(cfg.get("pool_timeout_seconds", connect_timeout))

        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )

        return ProviderBundle(
            client=client,
            model=cfg.get("model", "deepseek-chat"),
            thinking=bool(cfg.get("thinking", False)),
            max_tokens=int(cfg.get("max_tokens", 16384)),
        )
