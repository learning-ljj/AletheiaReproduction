"""Verifier agent wrapper with unified structured output contract."""

from __future__ import annotations

from typing import Callable

from src.core.pipeline import call_verifier as legacy_call_verifier
from src.core.state import VerificationDecision


class VerifierAgent:
    """Object-style verifier runtime.

    In C32 this wraps legacy verifier flow while enforcing the new output contract.
    """

    def __init__(
        self,
        *,
        llm_client,
        prompts: dict,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        verifier_runner: Callable | None = None,
    ):
        self.llm_client = llm_client
        self.prompts = prompts
        self.tools = tools
        self.tool_executor = tool_executor
        self._verifier_runner = verifier_runner or legacy_call_verifier

    @staticmethod
    def _ensure_tag(text: str, tag: str, default_content: str = "NONE") -> str:
        if f"<{tag}>" in text and f"</{tag}>" in text:
            return text
        suffix = f"\n<{tag}>{default_content}</{tag}>"
        return (text or "").rstrip() + suffix

    def run(
        self,
        *,
        problem_text: str,
        proof_text: str,
    ) -> tuple[str, VerificationDecision, str, list[dict], str]:
        full_text, decision, verification_report, tool_trace, phase1 = self._verifier_runner(
            self.llm_client,
            self.prompts,
            problem_text,
            proof_text,
            self.tools,
            self.tool_executor,
        )

        full_text = self._ensure_tag(full_text or "", "verified_lemmas", default_content="NONE")
        full_text = self._ensure_tag(full_text, "citation_review", default_content="NONE")

        return full_text, decision, verification_report, tool_trace, phase1
