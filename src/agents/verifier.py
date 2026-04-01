"""Verifier agent with object-style runtime and unified output contract."""

from __future__ import annotations

import json
import re
from typing import Callable

from src.agents.citation_reviewer import CitationReviewerAgent
from src.core.state import VerificationDecision
from src.memory.problem_memory import get_current_problem_memory
from src.utils.parser import (
    extract_preliminary_solution,
    parse_citations,
    extract_verification_report,
    parse_verification_decision,
)


class VerifierAgent:
    """Object-style verifier runtime (main-chain implementation)."""

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
        self._verifier_runner = verifier_runner

    @staticmethod
    def _has_xml_tag(text: str | None, tag: str) -> bool:
        if not text:
            return False
        return f"<{tag}>" in text and f"</{tag}>" in text

    @classmethod
    def _has_verifier_contract(cls, text: str | None) -> bool:
        return cls._has_xml_tag(text, "verdict") and cls._has_xml_tag(text, "verification")

    @staticmethod
    def _ensure_tag(text: str, tag: str, default_content: str = "NONE") -> str:
        if f"<{tag}>" in text and f"</{tag}>" in text:
            return text
        suffix = f"\n<{tag}>{default_content}</{tag}>"
        return (text or "").rstrip() + suffix

    @staticmethod
    def _upsert_tag(text: str, tag: str, content: str) -> str:
        pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
        replacement = f"<{tag}>{content}</{tag}>"
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)
        return (text or "").rstrip() + "\n" + replacement

    def _attach_citation_review(self, full_text: str, proof_text: str) -> str:
        cites = parse_citations(proof_text)
        if not cites:
            return self._ensure_tag(full_text, "citation_review", default_content="NONE")

        reviewer = CitationReviewerAgent(problem_memory=get_current_problem_memory())
        review = reviewer.review(cites=cites, claim_spans=[])
        return self._upsert_tag(
            full_text,
            "citation_review",
            json.dumps(review, ensure_ascii=False),
        )

    def _run_legacy_override(
        self,
        *,
        problem_text: str,
        proof_text: str,
    ) -> tuple[str, VerificationDecision, str, list[dict], str] | None:
        if self._verifier_runner is None:
            return None
        return self._verifier_runner(
            self.llm_client,
            self.prompts,
            problem_text,
            proof_text,
            self.tools,
            self.tool_executor,
        )

    def run(
        self,
        *,
        problem_text: str,
        proof_text: str,
    ) -> tuple[str, VerificationDecision, str, list[dict], str]:
        override = self._run_legacy_override(problem_text=problem_text, proof_text=proof_text)
        if override is not None:
            full_text, decision, verification_report, tool_trace, phase1 = override
            full_text = self._ensure_tag(full_text or "", "verified_lemmas", default_content="NONE")
            full_text = self._attach_citation_review(full_text, proof_text)
            return full_text, decision, verification_report, tool_trace, phase1

        solution_body = extract_preliminary_solution(proof_text)
        phase1_content = self.prompts["verifier"]["phase1_user"].format(
            problem_statement=problem_text,
            solution=solution_body,
        )

        messages: list = [
            {"role": "system", "content": self.prompts["verifier"]["system"]},
            {"role": "user", "content": phase1_content},
        ]

        phase1_resp = self.llm_client.chat(messages, thinking=True, stream_prefix="VERIFIER-P1")
        messages.append(
            {
                "role": "assistant",
                "content": phase1_resp.content or None,
                "reasoning_content": phase1_resp.reasoning_content or None,
            }
        )

        messages.append({"role": "user", "content": self.prompts["verifier"]["phase2_user"]})
        phase2_resp = self.llm_client.chat_with_tools(
            messages,
            self.tools,
            self.tool_executor,
            stream_prefix="VERIFIER-P2",
        )

        self.llm_client.clear_reasoning_content(messages)

        phase3_user_prompt = self.prompts["verifier"]["phase3_user"]
        phase3_retry_prompt = (
            phase3_user_prompt
            + "\n\nFORMAT REQUIRED:\n"
            + "<verdict>CORRECT|MINOR_FLAW|CRITICAL_FLAW</verdict>\n"
            + "<verification>完整验证报告</verification>\n"
            + "<verified_lemmas>...</verified_lemmas>\n"
            + "<citation_review>...</citation_review>"
        )
        messages.append({"role": "user", "content": phase3_user_prompt})
        phase3_resp = self.llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")

        full_text = phase3_resp.content or ""
        max_phase3_retries = 2
        if not self._has_verifier_contract(full_text):
            for _ in range(max_phase3_retries):
                messages[-1] = {"role": "user", "content": phase3_retry_prompt}
                phase3_resp = self.llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")
                full_text = phase3_resp.content or ""
                if self._has_verifier_contract(full_text):
                    break
            else:
                raise ValueError("Verifier Phase 3 missing required <verdict>/<verification> tags")

        full_text = self._ensure_tag(full_text, "verified_lemmas", default_content="NONE")
        full_text = self._attach_citation_review(full_text, proof_text)

        decision = parse_verification_decision(full_text)
        verification_report = "" if decision == VerificationDecision.CORRECT else extract_verification_report(full_text)
        phase1_analysis = phase1_resp.content or ""

        return full_text, decision, verification_report, phase2_resp.tool_calls_trace, phase1_analysis
