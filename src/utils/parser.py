"""输出解析器：从 LLM 输出中提取解答正文、boxed 答案、裁决等。"""

import logging
import re

from src.core.state import VerificationDecision

logger = logging.getLogger(__name__)

PRELIMINARY_SOLUTION_MARKER = "### Preliminary Solution ###"

# Bug #1 修复：兼容 LLM 输出的多种标题变体：
#   ### Preliminary Solution ###     （标准）
#   ### Preliminary Solution          （省略尾部 ###）
#   ## Preliminary Solution ##        （双井号）
#   ## Preliminary Solution           （双井号无尾）
#   **Preliminary Solution**          （Markdown 粗体）
#   Preliminary Solution:             （无 Markdown 前缀，仅冒号）
#   ### 2. Preliminary Solution ###   （LLM 擅自添加数字编号）
#   ### 2. Preliminary Solution       （带编号且省略尾部 ###）
_PRELIMINARY_SOLUTION_RE = re.compile(
    r"(?:#{2,3}\s*)?(?:\*\*)?(?:\d+\.\s*)?Preliminary\s+Solution(?:\*\*)?\s*(?:#{2,3})?",
    re.IGNORECASE,
)

# 裁决标签正则（旧格式）：兼容 Markdown 粗体（**[TAG]**）、前缀冒号（: [TAG]）、
# 空格分隔等各种 LLM 输出变体；不区分大小写。
_VERDICT_RE = re.compile(
    r"\[(?P<tag>CRITICAL_FLAW|MINOR_FLAW|CORRECT)\]",
    re.IGNORECASE,
)

# 新格式裁决正则：匹配 Phase 3 prompt 要求的 [DECISION]: VALUE 格式
# 例如: [DECISION]: CORRECT / [DECISION]: MINOR_FLAW / [DECISION]: CRITICAL_FLAW
_DECISION_TAG_RE = re.compile(
    r"\[DECISION\]\s*:\s*(?P<value>CRITICAL_FLAW|MINOR_FLAW|CORRECT)\b",
    re.IGNORECASE,
)


def extract_preliminary_solution(text: str) -> str:
    """提取 '### Preliminary Solution ###' 标记之后的完整解答正文。

    同时兼容 LLM 省略尾部 '###' 的情况（如 '### Preliminary Solution'）。
    如果找不到标记，发出警告日志并返回原始文本（容错处理，避免静默吞掉 Summary 区块）。
    """
    m = _PRELIMINARY_SOLUTION_RE.search(text)
    if m is None:
        logger.warning(
            "extract_preliminary_solution: '### Preliminary Solution ###' marker not found "
            "in LLM output (len=%d). Returning full text as fallback. "
            "This may cause Verifier to receive the Summary section as part of the solution.",
            len(text),
        )
        return text
    return text[m.end():].strip()


def extract_boxed_answer(text: str) -> str | None:
    r"""从文本中提取 \boxed{...} 中的最终答案/结论。

    支持嵌套大括号（如 \boxed{\frac{1}{2}}）。
    如果有多个 \boxed{}，返回最后一个。
    如果找不到，返回 None。
    """
    results: list[str] = []
    search_start = 0
    while True:
        idx = text.find(r"\boxed{", search_start)
        if idx == -1:
            break
        # 从 '{' 开始匹配嵌套大括号
        brace_start = idx + len(r"\boxed")
        depth = 0
        i = brace_start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[brace_start + 1 : i])
                    break
            i += 1
        search_start = i + 1 if i < len(text) else len(text)

    return results[-1] if results else None


def extract_section(text: str, marker: str, after: bool = True) -> str:
    """基于标记提取文本段落。

    Args:
        text: 完整文本
        marker: 分割标记字符串
        after: True 返回标记之后的文本，False 返回标记之前的文本
    Returns:
        提取的文本段落，找不到标记返回空字符串。
    """
    idx = text.find(marker)
    if idx == -1:
        return ""
    if after:
        return text[idx + len(marker) :].strip()
    else:
        return text[:idx].strip()


def extract_bug_report(verification_text: str) -> str:
    """从 Verifier 完整输出中提取 bug report（Summary 部分）。

    提取 'Detailed Verification' 标记之前的所有文本
    （包含 Summary + List of Findings）。
    如果找不到标记，返回完整文本。
    """
    result = extract_section(verification_text, "Detailed Verification", after=False)
    return result if result else verification_text


def parse_verification_verdict(verification_text: str) -> VerificationDecision:
    """从 Verifier 完整输出中解析三路裁决。

    解析策略（按优先级）：
      1a. 新格式：正则匹配 [DECISION]: VALUE（Phase 3 prompt 要求的强制格式）
            [DECISION]: CORRECT / [DECISION]: MINOR_FLAW / [DECISION]: CRITICAL_FLAW
      1b. 旧格式（兼容回退）：正则匹配所有 [TAG] 标签（大小写不敏感），
            按严重度优先：CRITICAL_FLAW > MINOR_FLAW > CORRECT
            兼容 Markdown 粗体包裹（**[CRITICAL_FLAW]**）等变体。
      2.  Fallback: 自由文本关键词搜索（'CRITICAL ERROR' / 'JUSTIFICATION GAP'）
      3.  全部未匹配 → 默认 MINOR_FLAW（保守策略），并发出警告日志
    """
    # 策略 1a：优先匹配新格式 [DECISION]: VALUE（Phase 3 prompt 要求的强制格式）
    m = _DECISION_TAG_RE.search(verification_text)
    if m:
        value = m.group("value").upper()
        if value == "CRITICAL_FLAW":
            return VerificationDecision.CRITICAL_FLAW
        if value == "MINOR_FLAW":
            return VerificationDecision.MINOR_FLAW
        if value == "CORRECT":
            return VerificationDecision.CORRECT

    # 策略 1b：回退到旧格式 [TAG]（兼容历史日志及 LLM 未严格遵守新格式的情况）
    # 正则提取所有 [TAG] 标签，按严重度优先返回
    found_tags = {tag.upper() for tag in _VERDICT_RE.findall(verification_text)}
    if "CRITICAL_FLAW" in found_tags:
        return VerificationDecision.CRITICAL_FLAW
    if "MINOR_FLAW" in found_tags:
        return VerificationDecision.MINOR_FLAW
    if "CORRECT" in found_tags:
        return VerificationDecision.CORRECT

    # 策略 2：自由文本关键词 Fallback
    text_upper = verification_text.upper()
    if "CRITICAL ERROR" in text_upper:
        logger.warning(
            "parse_verification_verdict: No [TAG] found; fallback to CRITICAL_FLAW "
            "via 'CRITICAL ERROR' keyword."
        )
        return VerificationDecision.CRITICAL_FLAW
    if "JUSTIFICATION GAP" in text_upper:
        logger.warning(
            "parse_verification_verdict: No [TAG] found; fallback to MINOR_FLAW "
            "via 'JUSTIFICATION GAP' keyword."
        )
        return VerificationDecision.MINOR_FLAW

    # 策略 3：默认保守策略（发出警告）
    logger.warning(
        "parse_verification_verdict: No verdict tag or keyword found in Verifier output "
        "(len=%d). Defaulting to MINOR_FLAW (conservative). "
        "Check if Verifier followed the Phase 3 output format.",
        len(verification_text),
    )
    return VerificationDecision.MINOR_FLAW
