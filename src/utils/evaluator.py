"""自动评分器：Benchmark 评测专用的解答评分工具。"""

import re

from src.utils.parser import extract_boxed_answer


def check_proof_completeness(predicted_proof: str) -> dict:
    """解答格式完整性检查（不依赖 LLM）。

    在 proofbench 评测时用于检查解答结构是否完整。
    当前规范下，完整解答以 `<solution>...</solution>` 为准。

    返回 {
        "has_preliminary_solution": bool,
        "has_summary": bool,
        "proof_length": int,
        "has_qed_marker": bool,
    }
    """
    text = predicted_proof or ""
    has_solution = bool(re.search(r"<solution>.*?</solution>", text, re.IGNORECASE | re.DOTALL))
    return {
        "has_preliminary_solution": has_solution,
        "has_summary": "Summary" in text,
        "proof_length": len(text),
        "has_qed_marker": bool(re.search(r"QED|Q\.E\.D\.|∎|□|\\blacksquare|\\square|\\qed", text)),
    }


def check_answer(predicted: str, ground_truth: str) -> bool:
    """短答题自动评分。

    从 predicted 中提取 \\boxed{} 答案，与 ground_truth 进行
    LaTeX 归一化比较。无 \\boxed{} 时尝试提取最后一行。
    """
    # 优先提取 \boxed{} 结果；若无则退化到最后一个非空行
    extracted = extract_boxed_answer(predicted or "")
    if extracted is None:
        lines = [l.strip() for l in (predicted or "").strip().splitlines() if l.strip()]
        extracted = lines[-1] if lines else ""

    return _normalize_latex(extracted) == _normalize_latex(ground_truth)


def _normalize_latex(s: str) -> str:
    """LaTeX 归一化：去空格、统一分数表示等，用于答案比较。"""
    s = s.strip()
    # 去除 $ 符号包裹
    s = s.strip("$")
    # 去除所有空格
    s = re.sub(r"\s+", "", s)
    # 统一 \dfrac → \frac
    s = s.replace(r"\dfrac", r"\frac")
    # 统一 \left( \right) → ( )
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    return s
