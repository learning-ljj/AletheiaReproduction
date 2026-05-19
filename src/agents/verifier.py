"""Verifier agent with object-style runtime and unified output contract."""

from __future__ import annotations
from typing import Callable

from src.utils.parsing.parser import extract_xml_tag


class VerifierAgent:
    """Object-style verifier runtime (main-chain implementation)."""

    def __init__(
        self,
        *,
        llm_client,
        prompts: dict,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 20,
    ):
        self.llm_client = llm_client
        self.prompts = prompts
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_rounds = max(1, int(max_rounds))

    def run(
        self,
        *,
        problem_text: str,
        proof_text: str,
    ) -> tuple[str, list[dict], str]:
        """运行三阶段验证并返回原始 LLM 输出。
        
        返回:
        - tuple(verifier_response, tool_trace, preliminary_analysis)
        其中 verifier_response 是 Phase 3 的完整输出，包含所有四个标签块。
        解析工作交由调用方（orchestrator）完成。
        """
        # 候选解答容错读取：
        # - 优先取 <solution> 正文；
        # - 若缺失，则把原始 proof_text 直接交给 Verifier 做“格式问题”审查。
        # 大白话：格式判定不在 Generator/Orchestrator 抢先做，统一由 Verifier 给出结论。
        raw_proof_text = (proof_text or "").strip()
        solution_body = extract_xml_tag(raw_proof_text, "solution").strip() or raw_proof_text

        # Phase1 负责整体分析和初步判断，输出给 Phase2 作为上下文提示。
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

        # Phase2 负责工具验证（run_python / read_artifact / review_citation）。
        messages.append({"role": "user", "content": self.prompts["verifier"]["phase2_user"]})
        phase2_resp = self.llm_client.chat_with_tools(
            messages,
            self.tools,
            self.tool_executor,
            max_rounds=self.max_rounds,
            stream_prefix="VERIFIER-P2",
        )

        # 新 user message 前清理历史 reasoning_content，避免占用上下文。
        self.llm_client.clear_reasoning_content(messages)

        # Phase3 负责综合判断和输出最终验证结论 + 验证报告 + 逐条引用审查。
        messages.append({"role": "user", "content": self.prompts["verifier"]["phase3_user"]})
        phase3_resp = self.llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")

        verifier_response = phase3_resp.content or ""
        tool_trace = getattr(phase2_resp, "tool_calls_trace", [])
        preliminary_analysis = phase1_resp.content or ""

        # 直接返回原始 LLM 输出，不做本地解析
        return verifier_response, tool_trace, preliminary_analysis
