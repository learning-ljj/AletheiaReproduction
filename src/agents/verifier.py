"""Verifier agent with object-style runtime and unified output contract."""

from __future__ import annotations

import json
import re
from typing import Callable

from src.agents.citation_reviewer import CitationReviewerAgent
from src.core.state import VerificationDecision
from src.memory.problem_memory import get_current_problem_memory
from src.tools.envelope import extract_tool_success_data, parse_tool_payload
from src.utils.parsing.parser import (
    extract_xml_tag,
    parse_citations_with_claim_spans,
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
        max_tool_rounds: int = 5,
    ):
        self.llm_client = llm_client
        self.prompts = prompts
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_tool_rounds = max(1, int(max_tool_rounds))

    @staticmethod
    def _has_xml_tag(text: str | None, tag: str) -> bool:
        # 契约校验：只要缺任意一个结束标签，上层就会触发格式重试。
        if not text:
            return False
        return f"<{tag}>" in text and f"</{tag}>" in text

    @classmethod
    def _has_verifier_contract(cls, text: str | None) -> bool:
        return cls._has_xml_tag(text, "verdict") and cls._has_xml_tag(text, "verification")

    @staticmethod
    def _ensure_tag(text: str, tag: str, default_content: str = "NONE") -> str:
        # 保底补标签：防止下游 parser 因字段缺失直接报错。
        # 大白话：LLM 漏写了字段，这里给它补个“占位值”。
        if f"<{tag}>" in text and f"</{tag}>" in text:
            return text
        suffix = f"\n<{tag}>{default_content}</{tag}>"
        return (text or "").rstrip() + suffix

    @staticmethod
    def _upsert_tag(text: str, tag: str, content: str) -> str:
        # upsert 语义：有就替换，没有就追加。
        # 作用：统一写入 citation_review / verified_lemmas，避免出现重复块。
        pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
        replacement = f"<{tag}>{content}</{tag}>"
        if pattern.search(text):
            return pattern.sub(lambda _: replacement, text, count=1)
        return (text or "").rstrip() + "\n" + replacement

    @staticmethod
    def _extract_citation_review_from_trace(tool_trace: list[dict] | None) -> dict | None:
        # 优先复用 Phase2 工具返回，避免重复跑 citation reviewer。
        # 倒序扫描是为了拿“最后一次”有效结果（最接近最终回答）。
        if not tool_trace:
            return None
        for item in reversed(tool_trace):
            if item.get("name") != "call_citation_reviewer":
                continue
            raw_result = item.get("result") or ""
            payload = parse_tool_payload(raw_result)
            review_payload = extract_tool_success_data(payload, allow_legacy=True)
            if not isinstance(review_payload, dict):
                continue

            if all(key in review_payload for key in ("summary", "items", "fail_count")):
                return {
                    "summary": review_payload.get("summary", ""),
                    "items": review_payload.get("items", []),
                    "fail_count": review_payload.get("fail_count", 0),
                }
        return None

    @staticmethod
    def _build_citation_phase2_prompt(cites: list[str], claim_spans: list[str]) -> str:
        if not cites:
            return "\n\nCitation Task: no [cite:path] markers were detected in the candidate solution."

        # 这里是“显式工具指令注入”：
        # 把 cites + claim_spans 原样给模型，强约束它调用 citation_reviewer。
        return (
            "\n\nCitation Task (required when cites exist):\n"
            "You detected citation markers in the candidate solution. "
            "Call tool `call_citation_reviewer` exactly once using these arrays, then use its result in Phase 3 <citation_review>.\n"
            f"cites={json.dumps(cites, ensure_ascii=False)}\n"
            f"claim_spans={json.dumps(claim_spans, ensure_ascii=False)}"
        )

    def _attach_citation_review(
        self,
        full_text: str,
        solution_text: str,
        tool_trace: list[dict] | None,
    ) -> str:
        # 注意：这里只解析 <solution>，不看外层思考文本。
        # 这样可以避免把思考区里的示例 [cite:xxx] 误当成正式引用。
        cites, claim_spans = parse_citations_with_claim_spans(solution_text)
        if not cites:
            return self._ensure_tag(full_text, "citation_review", default_content="NONE")

        review = self._extract_citation_review_from_trace(tool_trace)
        if review is None:
            # 双保险：如果 Phase2 没调工具，Verifier 自己补跑一次。
            # 目的不是“惩罚模型”，而是保证产物里一定有 citation_review 字段。
            reviewer = CitationReviewerAgent(problem_memory=get_current_problem_memory())
            review = reviewer.review(cites=cites, claim_spans=claim_spans)
        return self._upsert_tag(
            full_text,
            "citation_review",
            json.dumps(review, ensure_ascii=False),
        )

    def run(
        self,
        *,
        problem_text: str,
        proof_text: str,
    ) -> tuple[str, VerificationDecision, str, list[dict], str]:
        # 候选解答容错读取：
        # - 优先取 <solution> 正文；
        # - 若缺失，则把原始 proof_text 直接交给 Verifier 做“格式问题”审查。
        # 大白话：格式判定不在 Generator/Orchestrator 抢先做，统一由 Verifier 给出结论。
        raw_proof_text = (proof_text or "").strip()
        solution_body = extract_xml_tag(raw_proof_text, "solution").strip() or raw_proof_text

        # 从待审文本中提取 citation 与 claim span（用于后续 citation 审查）。
        cites, claim_spans = parse_citations_with_claim_spans(solution_body)
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

        # Phase2 负责工具验证（run_python / read_artifact_layer / call_citation_reviewer）。
        phase2_prompt = self.prompts["verifier"]["phase2_user"] + self._build_citation_phase2_prompt(cites, claim_spans)
        messages.append({"role": "user", "content": phase2_prompt})
        phase2_resp = self.llm_client.chat_with_tools(
            messages,
            self.tools,
            self.tool_executor,
            max_tool_rounds=self.max_tool_rounds,
            stream_prefix="VERIFIER-P2",
        )

        # 新 turn 前清理历史 reasoning_content，避免无意义上下文膨胀。
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
            # 格式自愈：当 Phase3 漏了关键 XML 标签，最多重试两次。
            for _ in range(max_phase3_retries):
                messages[-1] = {"role": "user", "content": phase3_retry_prompt}
                phase3_resp = self.llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")
                full_text = phase3_resp.content or ""
                if self._has_verifier_contract(full_text):
                    break
            else:
                raise ValueError("Verifier Phase 3 missing required <verdict>/<verification> tags")

        tool_trace = getattr(phase2_resp, "tool_calls_trace", [])

        # 统一补齐可选字段，避免下游“有时有、有时无”。
        full_text = self._ensure_tag(full_text, "verified_lemmas", default_content="NONE")
        full_text = self._attach_citation_review(full_text, solution_body, tool_trace)

        # 最终路由以 LLM verdict 为准（不再做本地规则覆写）。
        decision = parse_verification_decision(full_text)
        verification_report = "" if decision == VerificationDecision.CORRECT else extract_verification_report(full_text)

        phase1_analysis = phase1_resp.content or ""

        return full_text, decision, verification_report, tool_trace, phase1_analysis
