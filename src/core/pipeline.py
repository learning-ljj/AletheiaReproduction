"""推理流水线函数：Generator / Verifier / Reviser。"""

from typing import Callable

from src.core.state import VerificationDecision
from src.models.llm_client import LLMClient, LLMResponse
from src.utils.parser import (
    extract_generator_candidate_from_reasoning,
    extract_xml_tag,
    extract_preliminary_solution,
    extract_verification_report,
    parse_verification_decision,
)


def _has_xml_tag(text: str | None, tag: str) -> bool:
    """判断文本中是否包含完整的 XML 标签对。"""
    if not text:
        return False
    return f"<{tag}>" in text and f"</{tag}>" in text


def _has_verifier_contract(text: str | None) -> bool:
    """判断 Verifier 输出是否满足严格 XML 协议。"""
    return _has_xml_tag(text, "verdict") and _has_xml_tag(text, "verification")


def _has_final_assessor_contract(text: str | None) -> bool:
    """终局评估器必须输出 status+verdict+solution 三段。"""
    return _has_xml_tag(text, "status") and _has_xml_tag(text, "verdict") and _has_xml_tag(text, "solution")


def call_generator(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    lesson: str | None = None,
) -> LLMResponse:
    """调用 Generator 生成初始（或重试）解答。"""
    if lesson:
        user_content = (
            problem_text
            + "\n\n---\n**Note: Your previous attempt was rejected due to critical flaws. "
            "Avoid repeating the following mistakes in your new attempt:**\n\n"
            + lesson
        )
    else:
        user_content = problem_text

    _MAX_FORMAT_RETRIES = 2
    retry_hint = (
        "\n\n---\nFORMAT REQUIRED:\n"
        "Return non-empty content and include BOTH tags exactly once:\n"
        "<verdict>...</verdict>\n<solution>...</solution>"
    )

    last_response: LLMResponse | None = None
    for attempt in range(_MAX_FORMAT_RETRIES + 1):
        messages: list[dict] = [
            {"role": "system", "content": prompts["generator"]["system"]},
            {"role": "user", "content": user_content},
        ]
        response = llm_client.chat(messages, thinking=True, stream_prefix="GENERATOR")
        last_response = response

        if (
            (response.content or "").strip()
            and _has_xml_tag(response.content, "verdict")
            and _has_xml_tag(response.content, "solution")
        ):
            return response

        # 回退机制：content 缺失/不合规时，优先尝试从思维链中正则提取。
        extracted_from_reasoning = extract_generator_candidate_from_reasoning(
            response.reasoning_content or ""
        )
        if extracted_from_reasoning:
            print(
                "  [Generator] Recovered <verdict>/<solution> from reasoning_content, "
                "skip retry.",
                flush=True,
            )
            return LLMResponse(
                content=extracted_from_reasoning,
                reasoning_content=response.reasoning_content or "",
                tool_calls_trace=response.tool_calls_trace,
            )

        if attempt < _MAX_FORMAT_RETRIES:
            print(
                f"  [Generator] Missing required tags/content, retrying "
                f"({attempt + 1}/{_MAX_FORMAT_RETRIES})...",
                flush=True,
            )
            user_content = problem_text + retry_hint if not lesson else (
                problem_text
                + "\n\n---\n"
                + lesson
                + retry_hint
            )

    return last_response or LLMResponse(content="", reasoning_content="")


def call_verifier(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    proof_text: str,
    tool_schemas: list[dict],
    tool_executor: Callable[[str, dict], str],
) -> tuple[str, VerificationDecision, str, list[dict], str]:
    """调用 Verifier 验证解答（三阶段：初读->工具调用->汇总）。"""
    solution_body = extract_preliminary_solution(proof_text)
    phase1_content = prompts["verifier"]["phase1_user"].format(
        problem_statement=problem_text,
        solution=solution_body,
    )

    messages: list = [
        {"role": "system", "content": prompts["verifier"]["system"]},
        {"role": "user", "content": phase1_content},
    ]

    print("  [Verifier Phase 1] Initial analysis (thinking only)...", flush=True)
    phase1_resp = llm_client.chat(messages, thinking=True, stream_prefix="VERIFIER-P1")
    messages.append(
        {
            "role": "assistant",
            "content": phase1_resp.content or None,
            "reasoning_content": phase1_resp.reasoning_content or None,
        }
    )

    messages.append({"role": "user", "content": prompts["verifier"]["phase2_user"]})
    print("  [Verifier Phase 2] Tool verification (multi-turn)...", flush=True)
    phase2_resp = llm_client.chat_with_tools(
        messages,
        tool_schemas,
        tool_executor,
        stream_prefix="VERIFIER-P2",
    )

    llm_client.clear_reasoning_content(messages)

    phase3_user_prompt = prompts["verifier"]["phase3_user"]
    phase3_retry_prompt = (
        phase3_user_prompt
        + "\n\nFORMAT REQUIRED:\n"
        + "<verdict>CORRECT|MINOR_FLAW|CRITICAL_FLAW</verdict>\n"
        + "<verification>完整验证报告</verification>"
    )
    messages.append({"role": "user", "content": phase3_user_prompt})
    print("  [Verifier Phase 3] Consolidating verdict...", flush=True)
    phase3_resp = llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")

    full_text = phase3_resp.content or ""
    _MAX_PHASE3_RETRIES = 2
    if not _has_verifier_contract(full_text):
        for attempt in range(_MAX_PHASE3_RETRIES):
            print(
                f"  [Verifier Phase 3] Missing required XML contract (attempt {attempt + 1}/"
                f"{_MAX_PHASE3_RETRIES}), retrying...",
                flush=True,
            )
            messages[-1] = {"role": "user", "content": phase3_retry_prompt}
            phase3_resp = llm_client.chat(messages, thinking=False, stream_prefix="VERIFIER-P3")
            full_text = phase3_resp.content or ""
            if _has_verifier_contract(full_text):
                break
        else:
            raise ValueError("Verifier Phase 3 missing required <verdict>/<verification> tags")

    decision = parse_verification_decision(full_text)
    verification_report = "" if decision == VerificationDecision.CORRECT else extract_verification_report(full_text)
    phase1_analysis = phase1_resp.content or ""

    return full_text, decision, verification_report, phase2_resp.tool_calls_trace, phase1_analysis


def call_reviser(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    previous_solution: str,
    verification_report: str,
) -> LLMResponse:
    """调用 Reviser 根据 Verifier 反馈修正解答。"""
    _MAX_FORMAT_RETRIES = 2
    correction_instruction = prompts["reviser"]["correction_instruction"] + "\n\n" + verification_report
    retry_hint = (
        "\n\nFORMAT REQUIRED: Return non-empty content with EXACTLY one "
        "<solution>...</solution> block."
    )

    last_response: LLMResponse | None = None
    for attempt in range(_MAX_FORMAT_RETRIES + 1):
        messages: list[dict] = [
            {"role": "system", "content": prompts["reviser"]["system"]},
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": previous_solution},
            {
                "role": "user",
                "content": correction_instruction,
            },
        ]
        response = llm_client.chat(messages, thinking=True, stream_prefix="REVISER")
        last_response = response

        if (response.content or "").strip() and _has_xml_tag(response.content, "solution"):
            return response

        if attempt < _MAX_FORMAT_RETRIES:
            print(
                f"  [Reviser] Missing <solution> or empty content, retrying "
                f"({attempt + 1}/{_MAX_FORMAT_RETRIES})...",
                flush=True,
            )
            correction_instruction = (
                prompts["reviser"]["correction_instruction"]
                + "\n\n"
                + verification_report
                + retry_hint
            )

    return last_response or LLMResponse(content="", reasoning_content="")


def call_final_assessor(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    current_solution: str,
    last_verifier_decision: str,
    last_verification_report: str,
) -> tuple[str, str]:
    """在 verifier 轮次耗尽后，判定 PARTIAL_PROGRESS / BEYOND_CAPABILITY。"""
    user_content = prompts["final_assessor"]["user"].format(
        problem_statement=problem_text,
        current_solution=current_solution,
        last_verifier_decision=last_verifier_decision,
        last_verification_report=last_verification_report,
    )

    messages: list[dict] = [
        {"role": "system", "content": prompts["final_assessor"]["system"]},
        {"role": "user", "content": user_content},
    ]

    retry_user = user_content + (
        "\n\nFORMAT REQUIRED:\n"
        "<status>PARTIAL_PROGRESS|BEYOND_CAPABILITY</status>\n"
        "<verdict>...</verdict>\n"
        "<solution>...</solution>"
    )

    resp = llm_client.chat(messages, thinking=False, stream_prefix="FINAL-ASSESSOR")
    text = resp.content or ""
    if not _has_final_assessor_contract(text):
        messages[-1] = {"role": "user", "content": retry_user}
        resp = llm_client.chat(messages, thinking=False, stream_prefix="FINAL-ASSESSOR")
        text = resp.content or ""
        if not _has_final_assessor_contract(text):
            raise ValueError("Final assessor missing required <status>/<verdict>/<solution> tags")

    status = extract_xml_tag(text, "status").strip().upper()
    if status not in ("PARTIAL_PROGRESS", "BEYOND_CAPABILITY"):
        raise ValueError(f"Invalid final assessor status: {status!r}")
    return status, text
