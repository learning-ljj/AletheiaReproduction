"""输出解析器：从 LLM 输出中提取解答正文、boxed 答案、裁决等。"""

import re

from src.memory.state import VerificationDecision


class ParseContractError(ValueError):
	"""Structured parser error for routing decisions."""

	def __init__(self, parse_error_code: str, message: str, failure_reason: str | None = None):
		super().__init__(message)
		self.parse_error_code = parse_error_code
		self.failure_reason = failure_reason or parse_error_code


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


def extract_xml_tags(text: str, tag: str) -> list[str]:
	"""从文本中提取多个 <tag>...</tag> 块。"""
	if not text or not tag:
		return []
	pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
	return [(m.group(1) or "").strip() for m in pattern.finditer(text)]


def parse_decision(verification_text: str) -> VerificationDecision:
	"""从 Verifier 完整输出中解析三路裁决。"""
	verdict = extract_xml_tag(verification_text, "verdict")
	if verdict:
		# 有时 <verdict> 中会包含额外解释文字，或把关键词写在段落末尾。
		# 尝试在文本中查找已知的判定关键词（不区分大小写），优先取最后出现的关键词。
		verdict_text = re.findall(r"\b(CRITICAL_FLAW|MINOR_FLAW|CORRECT)\b", verdict, flags=re.IGNORECASE)
		if verdict_text:
			verdict_upper = verdict_text[-1].upper()
			if verdict_upper == VerificationDecision.CRITICAL_FLAW.value:
				return VerificationDecision.CRITICAL_FLAW
			if verdict_upper == VerificationDecision.MINOR_FLAW.value:
				return VerificationDecision.MINOR_FLAW
			if verdict_upper == VerificationDecision.CORRECT.value:
				return VerificationDecision.CORRECT

		raise ParseContractError(
			"invalid_verdict",
			f"Invalid <verdict> value: {verdict!r}",
		)

	raise ParseContractError(
		"invalid_verdict",
		"Missing <verdict> tag in Verifier output",
	)


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
