"""最终输出构造器：统一成功、部分进展与失败场景的最终文案。"""

from src.utils.parser import extract_xml_tag


def build_final_output(
    success: bool,
    solution_text: str | None,
    failure_reason: str | None,
    *,
    partial: bool = False,
    assessment_output: str | None = None,
) -> str:
    """构造最终输出文本。

    Args:
        success: 是否完整正确解答（Verifier 判定 CORRECT）。
        solution_text: 解答内容（SUCCESS / PARTIAL 时非空）。
        failure_reason: 失败原因标识（FAILED 时非空）。
        partial: 是否为部分进展（轮次耗尽但有实质性解答内容）。
        assessment_output: Final Assessor 的原始 XML 输出（可选）。
    """
    if success:
        return (solution_text or "").strip()

    if assessment_output:
        solution_block = extract_xml_tag(assessment_output, "solution").strip()
        verdict_block = extract_xml_tag(assessment_output, "verdict").strip()
        status_block = extract_xml_tag(assessment_output, "status").strip().upper()
        if solution_block:
            return solution_block
        if verdict_block:
            return verdict_block
        if status_block == "BEYOND_CAPABILITY":
            return "Admits failure: beyond_capability."

    if partial and solution_text:
        return (solution_text or "").strip()

    reason = (failure_reason or "unknown_reason").strip() or "unknown_reason"
    return f"Admits failure: {reason}."
