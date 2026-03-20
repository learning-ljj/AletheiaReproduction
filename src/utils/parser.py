"""输出解析器：从 LLM 输出中提取解答正文、boxed 答案、裁决等。"""

import re

from src.core.state import VerificationDecision


def extract_xml_tag(text: str, tag: str) -> str:
    """从文本中提取 <tag>...</tag> 内的内容；找不到返回空字符串。"""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = text.find(open_tag)
    if start == -1:
        return ""
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def extract_preliminary_solution(text: str) -> str:
    """严格提取 `<solution>...</solution>` 内的完整解答正文。"""
    solution = extract_xml_tag(text, "solution")
    if solution:
        return solution
    raise ValueError("Missing <solution> tag")


def extract_generator_candidate_from_reasoning(reasoning_text: str) -> str:
    """从 Generator 思维链中正则提取并标准化 `<verdict>/<solution>`。

    仅当两个标签都能提取到非空内容时返回标准化文本；
    否则返回空字符串。
    """
    text = reasoning_text or ""
    if not text.strip():
        return ""

    # 严格匹配：要求小写标签 <verdict>...</verdict> 紧跟可选空白后 <solution>...</solution>
    # 仅允许空格、制表或换行作为两标签之间的间隔；不支持大小写变体。
    pair_re = re.compile(r"<verdict>(.*?)</verdict>[ \t\r\n]*<solution>(.*?)</solution>", re.DOTALL)
    matches = list(pair_re.finditer(text))
    if not matches:
        return ""

    last = matches[-1]
    verdict = (last.group(1) or "").strip()
    solution = (last.group(2) or "").strip()
    if not verdict or not solution:
        return ""

    return f"<verdict>{verdict}</verdict>\n<solution>{solution}</solution>"


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


def extract_verification_report(verification_text: str) -> str:
    """严格从 `<verification>...</verification>` 中提取验证报告。"""
    result = extract_xml_tag(verification_text, "verification")
    if result:
        return result
    raise ValueError("Missing <verification> tag in Verifier output")


def parse_verification_decision(verification_text: str) -> VerificationDecision:
    """从 Verifier 完整输出中解析三路裁决。"""
    verdict_text = extract_xml_tag(verification_text, "verdict")
    if verdict_text:
        verdict_upper = verdict_text.strip().upper()
        if verdict_upper == VerificationDecision.CRITICAL_FLAW.value:
            return VerificationDecision.CRITICAL_FLAW
        if verdict_upper == VerificationDecision.MINOR_FLAW.value:
            return VerificationDecision.MINOR_FLAW
        if verdict_upper == VerificationDecision.CORRECT.value:
            return VerificationDecision.CORRECT
        raise ValueError(f"Invalid <verdict> value: {verdict_text!r}")

    raise ValueError("Missing <verdict> tag in Verifier output")


def normalize_short_answer(text: str) -> str:
    """Normalize a short-answer string for exact-match checking.

    Goal: extract a concise canonical representation for short answers
    (prefer integers/fractions/decimal) while stripping LaTeX tags,
    surrounding words like '答案', and punctuation.
    """
    if not text:
        return ""
    import re

    s = str(text).strip()
    # Remove XML/HTML tags
    s = re.sub(r"<[^>]+>", "", s)
    # Replace full-width digits with ASCII
    s = s.translate(str.maketrans(
        {chr(0xFF10 + i): str(i) for i in range(10)}
    ))
    # Remove common leading labels
    s = re.sub(r'(?i)^(答案[:：\s]*|answer[:：\s]*|verdict[:：\s]*)', "", s).strip()
    # If inside $...$ math, unwrap the first math region
    m = re.search(r"\$(.*?)\$", s)
    if m:
        s = m.group(1).strip()

    # Try to find an integer, fraction or decimal number first
    m = re.search(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", s)
    if m:
        return m.group(0).strip()

    # Fallback: remove surrounding punctuation and whitespace
    s = s.strip().strip('.,;:\\"\'()[]')
    return s
