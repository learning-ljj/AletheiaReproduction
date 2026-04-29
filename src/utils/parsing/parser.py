"""输出解析器：从 LLM 输出中提取解答正文、boxed 答案、裁决等。"""

import re

from src.memory.state import VerificationDecision


class ParseContractError(ValueError):
	"""Structured parser error for routing decisions."""

	def __init__(self, parse_error_code: str, message: str, failure_reason: str | None = None):
		super().__init__(message)
		self.parse_error_code = parse_error_code
		self.failure_reason = failure_reason or parse_error_code


def classify_parse_error(raw_text: str | None, exc: Exception) -> tuple[str, str]:
	"""Classify parse exceptions into stable route-friendly error codes."""
	if isinstance(exc, ParseContractError):
		return exc.parse_error_code, exc.failure_reason

	msg = (str(exc) or "").lower()
	text = (raw_text or "").lower()

	if "missing <solution>" in msg or ("<solution>" in text and "</solution>" not in text):
		return "missing_solution", "missing_solution"

	if "invalid <verdict>" in msg or "missing <verdict>" in msg:
		return "invalid_verdict", "invalid_verdict"

	if "missing <verification>" in msg:
		return "malformed_tag", "malformed_tag"

	if "missing required" in msg or "missing <status>" in msg:
		return "malformed_tag", "malformed_tag"

	return "malformed_tag", "malformed_tag"


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


def parse_lemmas(text: str) -> list[str]:
	"""提取所有 <lemma> 块内容。"""
	return extract_xml_tags(text, "lemma")


def parse_verified_lemmas(text: str) -> list[str]:
	"""提取所有 <verified_lemmas> 块内容。"""
	return extract_xml_tags(text, "verified_lemmas")


def parse_citations(text: str) -> list[str]:
	"""提取 [cite:path] 引用路径。"""
	if not text:
		return []

	citations: list[str] = []
	pattern = re.compile(r"\[cite:([^\]]*)\]")
	for match in pattern.finditer(text):
		raw_path = (match.group(1) or "").strip()
		if not raw_path:
			raise ValueError("Malformed citation block: empty [cite:path] payload")
		citations.append(raw_path)
	return citations


def _extract_claim_span(text: str, start: int, end: int) -> str:
	left_markers = ["\n", ".", "!", "?", "。", "！", "？"]
	right_markers = ["\n", ".", "!", "?", "。", "！", "？"]

	left_idx = -1
	for marker in left_markers:
		idx = text.rfind(marker, 0, start)
		if idx > left_idx:
			left_idx = idx
	left = left_idx + 1 if left_idx >= 0 else 0

	right = len(text)
	for marker in right_markers:
		idx = text.find(marker, end)
		if idx != -1:
			right = min(right, idx + 1)

	span = text[left:right]
	span = span.replace(text[start:end], " ")
	span = re.sub(r"<[^>]+>", " ", span)
	span = " ".join(span.split())
	if span:
		return span

	fallback_left = max(0, start - 80)
	fallback_right = min(len(text), end + 80)
	fallback = text[fallback_left:fallback_right].replace(text[start:end], " ")
	fallback = re.sub(r"<[^>]+>", " ", fallback)
	return " ".join(fallback.split())


def parse_citations_with_claim_spans(text: str) -> tuple[list[str], list[str]]:
	"""提取 [cite:path] 及其所在句子的 claim span。"""
	if not text:
		return [], []

	citations: list[str] = []
	claim_spans: list[str] = []
	pattern = re.compile(r"\[cite:([^\]]*)\]")
	for match in pattern.finditer(text):
		raw_path = (match.group(1) or "").strip()
		if not raw_path:
			raise ValueError("Malformed citation block: empty [cite:path] payload")
		citations.append(raw_path)
		claim_spans.append(_extract_claim_span(text, *match.span()))
	return citations, claim_spans


def parse_citation_review(text: str) -> str:
	"""提取 <citation_review> 块内容。"""
	return extract_xml_tag(text, "citation_review")


def parse_protocol_blocks(text: str) -> dict[str, object]:
	"""一次性解析 B20 协议字段。"""
	return {
		"lemmas": parse_lemmas(text),
		"verified_lemmas": parse_verified_lemmas(text),
		"citations": parse_citations(text),
		"citation_review": parse_citation_review(text),
	}


def extract_preliminary_solution(text: str) -> str:
	"""严格提取 `<solution>...</solution>` 内的完整解答正文。"""
	solution = extract_xml_tag(text, "solution")
	if solution:
		return solution
	raise ParseContractError("missing_solution", "Missing <solution> tag")


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
	raise ParseContractError(
		"malformed_tag",
		"Missing <verification> tag in Verifier output",
	)


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
		raise ParseContractError(
			"invalid_verdict",
			f"Invalid <verdict> value: {verdict_text!r}",
		)

	raise ParseContractError(
		"invalid_verdict",
		"Missing <verdict> tag in Verifier output",
	)

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
