"""Provider selection and OpenAI-compatible client creation helpers."""

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
    stream_max_retries: int
    stream_retry_backoff_seconds: float


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
        shared_defaults = config.get("llm_defaults") or {}
        provider_config = config.get(provider_name) or {}
        return {**shared_defaults, **provider_config}

    @classmethod
    def resolve_provider_payload(cls, config: dict) -> dict:
        """Resolve provider from full settings and normalize to {'deepseek': cfg}."""
        provider = config.get("provider")
        if not cls._configured(provider):
            volcano_api_key = (config.get("volcano") or {}).get("api_key")
            deepseek_api_key = (config.get("deepseek") or {}).get("api_key")
            if cls._configured(volcano_api_key):
                provider = "volcano"
            elif cls._configured(deepseek_api_key):
                provider = "deepseek"
            else:
                provider = "deepseek"

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
            return {"deepseek": volcano_cfg}

        if provider == "deepseek":
            deep_cfg = cls._resolve_provider_config(config, "deepseek")
            api_key = deep_cfg.get("api_key")
            base_url = deep_cfg.get("base_url")
            if not api_key or cls._contains_placeholder(api_key):
                raise ValueError(
                    "DeepSeek provider selected but `deepseek.api_key` is missing or contains placeholder. "
                    "Ensure .env contains DEEPSEEK_API_KEY and that you called load_dotenv() before loading config."
                )
            if base_url and not str(base_url).startswith("http"):
                raise ValueError(
                    f"DeepSeek base_url looks invalid: {base_url!r}. It must start with 'http://' or 'https://'."
                )
            return {"deepseek": deep_cfg}

        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'deepseek' or 'volcano'.")

    @staticmethod
    def build_client_bundle(provider_payload: dict) -> ProviderBundle:
        """Create OpenAI SDK client and stream/runtime options from normalized payload."""
        ds = provider_payload["deepseek"]
        connect_timeout = float(ds.get("connect_timeout_seconds", 30.0))
        read_timeout = float(ds.get("read_timeout_seconds", 90.0))
        write_timeout = float(ds.get("write_timeout_seconds", read_timeout))
        pool_timeout = float(ds.get("pool_timeout_seconds", connect_timeout))

        client = OpenAI(
            api_key=ds["api_key"],
            base_url=ds["base_url"],
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
        )

        return ProviderBundle(
            client=client,
            model=ds.get("model", "deepseek-chat"),
            thinking=bool(ds.get("thinking", False)),
            max_tokens=int(ds.get("max_tokens", 16384)),
            stream_max_retries=max(int(ds.get("stream_max_retries", 2)), 0),
            stream_retry_backoff_seconds=max(float(ds.get("stream_retry_backoff_seconds", 2.0)), 0.0),
        )
