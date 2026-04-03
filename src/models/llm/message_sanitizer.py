"""Utilities for sanitizing LLM conversation messages."""


class MessageSanitizer:
    """Message sanitization helpers shared by LLM runtime components."""

    @staticmethod
    def clear_reasoning_content(messages: list) -> None:
        """Clear reasoning_content from assistant messages in-place."""
        for message in messages:
            if isinstance(message, dict):
                if "reasoning_content" in message:
                    message["reasoning_content"] = None
            elif hasattr(message, "reasoning_content"):
                message.reasoning_content = None
