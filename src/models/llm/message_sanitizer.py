"""Utilities for sanitizing LLM conversation messages."""


class MessageSanitizer:
    """Message sanitization helpers shared by LLM runtime components."""

    @staticmethod
    def clear_reasoning_content(messages: list[dict]) -> None:
        """Clear reasoning_content from assistant messages in-place.

        当前主链统一使用 dict 形式消息，这里不再保留 SDK 对象兼容分支，
        以减少分支复杂度并避免误修改非消息对象。
        """
        for message in messages:
            if isinstance(message, dict) and "reasoning_content" in message:
                message["reasoning_content"] = None
