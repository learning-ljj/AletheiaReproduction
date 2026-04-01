"""最终输出构造器：统一成功、部分进展与失败场景的最终文案。"""

from src.utils.parser import extract_xml_tag


def build_final_output(
    success: bool,
    solution_text: str | None,
    failure_reason: str | None,
    *,
    partial: bool = False,
    assessment_output: str | None = None,
    preserve_xml: bool = False,
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
    base_output = ""

    if success:
        base_output = (solution_text or "").strip()
    elif assessment_output:
        if preserve_xml:
            base_output = assessment_output.strip()
        else:
            solution_block = extract_xml_tag(assessment_output, "solution").strip()
            verdict_block = extract_xml_tag(assessment_output, "verdict").strip()
            status_block = extract_xml_tag(assessment_output, "status").strip().upper()
            if solution_block:
                base_output = solution_block
            elif verdict_block:
                base_output = verdict_block
            elif status_block == "BEYOND_CAPABILITY":
                base_output = "Admits failure: beyond_capability."
    elif partial and solution_text:
        base_output = (solution_text or "").strip()
    else:
        reason = (failure_reason or "unknown_reason").strip() or "unknown_reason"
        base_output = f"Admits failure: {reason}."

    warning_text = (warning_summary or "").strip()
    if warning_text:
        return (base_output + "\n\n## Citation Warnings\n" + warning_text).strip()
    return base_output
