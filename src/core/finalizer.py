"""最终输出构造器：统一成功、部分进展与失败场景的最终文案。"""

from src.models.llm_client import LLMClient
from src.utils.parsing.parser import extract_xml_tag, parse_final_xml_output


def _has_xml_tag(text: str | None, tag: str) -> bool:
    """判断文本中是否包含完整的 XML 标签对。"""
    if not text:
        return False
    return f"<{tag}>" in text and f"</{tag}>" in text


def _has_final_contract(text: str | None) -> bool:
    """FINAL 必须输出 status+verdict+solution 三段。"""
    return _has_xml_tag(text, "status") and _has_xml_tag(text, "verdict") and _has_xml_tag(text, "solution")


def call_final(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    current_solution: str,
    last_verifier_decision: str,
    last_verification_report: str,
) -> tuple[str, str, str, str]:
    """在 verifier 轮次耗尽后，单次调用 FINAL 判定终态并给出最终报告。"""
    final_prompts = prompts.get("final")
    if not final_prompts:
        raise ValueError("Missing prompts.final")

    user_content = final_prompts["user"].format(
        problem_statement=problem_text,
        current_solution=current_solution,
        last_verifier_decision=last_verifier_decision,
        last_verification_report=last_verification_report,
    )

    messages: list[dict] = [
        {"role": "system", "content": final_prompts["system"]},
        {"role": "user", "content": user_content},
    ]

    retry_user = user_content + (
        "\n\nFORMAT REQUIRED:\n"
        "<status>PARTIAL_PROGRESS|BEYOND_CAPABILITY</status>\n"
        "<verdict>...</verdict>\n"
        "<solution>...</solution>"
    )

    resp = llm_client.chat(messages, thinking=False, stream_prefix="FINAL")
    text = resp.content or ""
    if not _has_final_contract(text):
        messages[-1] = {"role": "user", "content": retry_user}
        resp = llm_client.chat(messages, thinking=False, stream_prefix="FINAL")
        text = resp.content or ""
        if not _has_final_contract(text):
            raise ValueError("FINAL output missing required <status>/<verdict>/<solution> tags")

    status, verdict, solution = parse_final_xml_output(text)
    return status, verdict, solution, text


def build_final_output(
    success: bool,
    solution_text: str | None,
    failure_reason: str | None,
    *,
    partial: bool = False,
    assessment_output: str | None = None,
    preserve_xml: bool = False,
    references: list[str] | None = None,
    warning_summary: str | None = None,
) -> str:
    """构造最终输出文本。

    Args:
        success: 是否完整正确解答（Verifier 判定 CORRECT）。
        solution_text: 解答内容（SUCCESS / PARTIAL 时非空）。
        failure_reason: 失败原因标识（FAILED 时非空）。
        partial: 是否为部分进展（轮次耗尽但有实质性解答内容）。
        assessment_output: FINAL 节点的原始 XML 输出（可选）。
        preserve_xml: 若为 True，优先保留原始 XML 输出，便于后续结构化处理。
    """
    reason = (failure_reason or "unknown_reason").strip() or "unknown_reason"
    solution_block = extract_xml_tag(assessment_output or "", "solution").strip()
    verdict_block = extract_xml_tag(assessment_output or "", "verdict").strip()
    status_block = extract_xml_tag(assessment_output or "", "status").strip().upper()

    candidates = [
        (solution_text or "").strip() if success else "",
        (assessment_output or "").strip() if preserve_xml else "",
        solution_block,
        verdict_block,
        (solution_text or "").strip() if partial else "",
        "Admits failure: beyond_capability." if status_block == "BEYOND_CAPABILITY" else "",
        f"Admits failure: {reason}.",
    ]

    base_output = next((item for item in candidates if item), f"Admits failure: {reason}.")

    output = base_output.strip()

    reference_items = references or []
    if reference_items:
        output = output + "\n\n## References\n" + "\n".join(reference_items)

    warning_text = (warning_summary or "").strip()
    if warning_text:
        output = output + "\n\n## Citation Warnings\n" + warning_text

    return output.strip()
