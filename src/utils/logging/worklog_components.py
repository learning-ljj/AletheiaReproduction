"""Composable worklog building components."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime

from src.agents.worklog_summary import WorklogSummaryAgent
from src.core.config import load_prompts
from src.models.llm_client import create_llm_client


class EventCollector:
    """Collect run-level metadata and turn-indexed event buckets."""

    @staticmethod
    def _parse_ts(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    def collect(self, *, problem_id: str, events: list[dict]) -> dict:
        # 先把事件时间戳转成 datetime，用于计算总耗时。
        # 大白话：日志里每条事件都带 timestamp，这里取最早和最晚的差值当“整次运行耗时”。
        timestamps = [self._parse_ts(event.get("timestamp")) for event in events]
        timestamps = [item for item in timestamps if item is not None]
        elapsed = 0.0
        if len(timestamps) >= 2:
            elapsed = (max(timestamps) - min(timestamps)).total_seconds()

        # 迭代轮次不是直接读 state，而是从事件里的 turn_id 反推最大值。
        # 大白话：这样即使中间崩了，仍能从已落盘事件恢复“跑到第几轮”。
        tracked_nodes = {"GENERATOR", "VERIFIER", "REVISER"}
        turn_ids = [
            int(event.get("turn_id", 0))
            for event in events
            if str(event.get("agent_node")) in tracked_nodes
        ]
        iteration_count = max(turn_ids) if turn_ids else 0

        # 问题文本和标准答案优先从 RUN_START 提取。
        # 若字段结构变体（input.problem_text），这里也做兼容兜底。
        first_event = events[0] if events else {}
        problem_text = first_event.get("problem_text") or first_event.get("input", {}).get("problem_text") or ""
        ground_truth = first_event.get("ground_truth") or ""

        # 最终输出取最后一个 FINAL 事件，保证拿到最终版本而不是中间草稿。
        final_event = next((event for event in reversed(events) if event.get("agent_node") == "FINAL"), {})
        final_output = final_event.get("final_output") or final_event.get("content") or ""

        # 按 turn_id 分桶，后续渲染时按轮次回放“谁在第几轮说了什么”。
        turns: dict[int, list[dict]] = {}
        for event in events:
            node = str(event.get("agent_node", ""))
            if node not in tracked_nodes:
                continue
            turn_id = int(event.get("turn_id", 0))
            turns.setdefault(turn_id, []).append(event)

        return {
            "problem_id": problem_id,
            "elapsed": elapsed,
            "iteration_count": iteration_count,
            "problem_text": problem_text,
            "ground_truth": ground_truth,
            "final_output": final_output,
            "turns": turns,
        }


class SummaryLLMService:
    """Manage dedicated LLM session and JSON-only prompt execution for worklog summaries."""

    def __init__(self, *, llm_client=None, llm_config: dict | None = None, prompts: dict | None = None):
        self.llm_client = llm_client
        self.llm_config = llm_config
        self._active_llm_client = None
        self._active_llm_error = None
        self._agent: WorklogSummaryAgent | None = None
        self._llm_timeout_seconds = self._resolve_llm_timeout_seconds(llm_config)
        merged_prompts = prompts if isinstance(prompts, dict) else load_prompts()
        self._worklog_prompts = merged_prompts.get("worklog", {}) if isinstance(merged_prompts, dict) else {}

    @staticmethod
    def _resolve_llm_timeout_seconds(llm_config: dict | None) -> float:
        default_timeout = 120.0
        if not isinstance(llm_config, dict):
            return default_timeout
        agent_cfg = llm_config.get("agent") or {}
        timeout = agent_cfg.get("worklog_llm_timeout_seconds", default_timeout)
        try:
            timeout = float(timeout)
            return timeout if timeout > 0 else default_timeout
        except (TypeError, ValueError):
            return default_timeout

    @staticmethod
    def _extract_json(text: str) -> dict:
        payload = (text or "").strip()
        if payload.startswith("```"):
            lines = payload.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                payload = "\n".join(lines[1:-1]).strip()
        return json.loads(payload)

    def begin_session(self) -> None:
        # 每次 build_worklog 前都开一段“干净会话”。
        # 大白话：清空上次失败状态，避免旧错误污染本次摘要结果。
        self._active_llm_error = None
        self._agent = None

        if self.llm_config is not None:
            try:
                self._active_llm_client = create_llm_client(self.llm_config)
            except Exception as exc:  # noqa: BLE001
                self._active_llm_client = None
                self._active_llm_error = f"llm_client_init_exception:{type(exc).__name__}:{exc}"
                return
        else:
            self._active_llm_client = self.llm_client

        # 没有可用 LLM 时直接进入“可解释失败”模式，后续 request_json 会返回错误码。
        if self._active_llm_client is None:
            return

        system_prompt = self._worklog_prompts.get("summary_system") or "你是审阅器，只输出 JSON。"
        self._agent = WorklogSummaryAgent(
            llm_client=self._active_llm_client,
            system_prompt=system_prompt,
        )

    def end_session(self) -> None:
        self._active_llm_client = None
        self._active_llm_error = None
        self._agent = None

    def request_json(
        self,
        *,
        template_key: str,
        template_values: dict,
        required_keys: list[str],
    ) -> tuple[dict | None, str | None]:
        # 这是 worklog 摘要链路最核心的“结构化调用门面”：
        # 1) 渲染 prompt
        # 2) 调 LLM
        # 3) 校验 JSON 和必须字段
        # 4) 失败时返回机器可读错误码，不抛散乱异常
        if self._agent is None:
            if self._active_llm_error:
                return None, self._active_llm_error
            return None, "llm_client_not_configured"

        template = self._worklog_prompts.get(template_key)
        if not template:
            return None, f"prompt_template_missing:{template_key}"

        try:
            user_prompt = template.format(**template_values)
        except KeyError as exc:
            return None, f"prompt_template_key_missing:{template_key}:{exc}"

        holder: dict[str, object] = {}

        def _run() -> None:
            try:
                holder["resp"] = self._agent.run_summary(user_prompt)
            except Exception as exc:  # noqa: BLE001
                holder["exc"] = exc

        # 用线程 + join(timeout) 给 LLM 调用加硬超时。
        # 大白话：防止模型卡住把整条 worklog 生成流程拖死。
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=self._llm_timeout_seconds)
        if worker.is_alive():
            return None, f"llm_timeout:{self._llm_timeout_seconds:.1f}s"

        if "exc" in holder:
            exc = holder["exc"]
            return None, f"llm_exception:{type(exc).__name__}:{exc}"

        try:
            resp = holder.get("resp")
            if resp is None:
                return None, "llm_missing_response"
            parsed = self._extract_json(resp.content or "")
            for key in required_keys:
                if key not in parsed:
                    return None, f"llm_missing_key:{key}"
            return parsed, None
        except Exception as exc:  # noqa: BLE001
            return None, f"llm_exception:{type(exc).__name__}:{exc}"


class RoleSummarizer:
    """Generate role/tool summaries with structured fallbacks."""

    def __init__(self, summary_service: SummaryLLMService):
        self.summary_service = summary_service

    @staticmethod
    def _clip(text: str, max_chars: int = 3200) -> str:
        payload = (text or "").strip()
        if len(payload) <= max_chars:
            return payload
        return payload[:max_chars] + f"\n...[truncated {len(payload) - max_chars} chars]"

    @staticmethod
    def _as_text_list(value) -> list[str] | None:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return None

    @staticmethod
    def strip_leading_enumeration(text: str) -> str:
        payload = (text or "").strip()
        if not payload:
            return payload

        patterns = [
            r"^\(?\d+\)?[\.)、．]\s*",
            r"^[（(]\d+[）)]\s*",
            r"^[一二三四五六七八九十]+[、.．]\s*",
        ]
        for pattern in patterns:
            payload = re.sub(pattern, "", payload)
        return payload.strip()

    def summarize_reasoning(self, role: str, raw_cot: str) -> dict:
        cot = (raw_cot or "").strip()
        if not cot:
            return {
                "step_summary": ["(empty)"],
                "quality_evaluation": ["思维链为空，无法评估。"],
                "llm_fallback": True,
                "llm_error": "empty_reasoning",
            }

        parsed, err = self.summary_service.request_json(
            template_key="reasoning_user_template",
            template_values={
                "role": role,
                "reasoning_text": self._clip(cot),
            },
            required_keys=["step_summary", "quality_evaluation"],
        )
        if parsed is not None:
            step_summary = self._as_text_list(parsed.get("step_summary"))
            quality_evaluation = self._as_text_list(parsed.get("quality_evaluation"))
            if step_summary is not None and quality_evaluation is not None:
                return {
                    "step_summary": step_summary,
                    "quality_evaluation": quality_evaluation,
                    "llm_fallback": False,
                    "llm_error": None,
                }

        # 兜底策略：LLM 摘要失败时，仍返回结构化占位结果。
        # 大白话：宁可“有保底信息”，也不能让渲染器因为 None 崩掉。
        return {
            "step_summary": [f"{role} 思维链已记录（{len(cot)} 字符）。", "LLM 离线摘要失败，建议人工复核。"],
            "quality_evaluation": ["回退模式：未能完成细粒度质量评估。"],
            "llm_fallback": True,
            "llm_error": err or "llm_invalid_payload",
        }

    def summarize_role_content(self, role: str, content: str) -> dict:
        text = (content or "").strip()
        if not text:
            return {
                "content_summary": ["(empty)"],
                "content_quality": ["content 为空，无法评估。"],
                "llm_fallback": True,
                "llm_error": "empty_content",
            }

        parsed, err = self.summary_service.request_json(
            template_key="content_user_template",
            template_values={
                "role": role,
                "content_text": self._clip(text),
            },
            required_keys=["content_summary", "content_quality"],
        )
        if parsed is not None:
            content_summary = self._as_text_list(parsed.get("content_summary"))
            content_quality = self._as_text_list(parsed.get("content_quality"))
            if content_summary is not None and content_quality is not None:
                return {
                    "content_summary": content_summary,
                    "content_quality": content_quality,
                    "llm_fallback": False,
                    "llm_error": None,
                }

        return {
            "content_summary": ["LLM 离线摘要失败，回退到规则占位。"],
            "content_quality": ["回退模式：建议人工核查答案完整性与计算正确性。"],
            "llm_fallback": True,
            "llm_error": err or "llm_invalid_payload",
        }

    def summarize_verifier_phase2_tools(self, tool_calls: list[dict]) -> list[dict]:
        summaries: list[dict] = []
        for call in (tool_calls or []):
            name = str(call.get("name", ""))
            arguments = call.get("arguments", {}) or {}
            result = str(call.get("result", "") or "")

            parsed, err = self.summary_service.request_json(
                template_key="tool_user_template",
                template_values={
                    "tool_name": name,
                    "tool_arguments": json.dumps(arguments, ensure_ascii=False),
                    "tool_result": self._clip(result, max_chars=1800),
                },
                required_keys=[
                    "purpose",
                    "input_reasonableness",
                    "result_core",
                    "impact_on_verdict",
                    "process_audit",
                ],
            )

            if parsed is not None:
                summaries.append(
                    {
                        "name": name,
                        "purpose": str(parsed["purpose"]),
                        "input_reasonableness": str(parsed["input_reasonableness"]),
                        "result_core": str(parsed["result_core"]),
                        "impact_on_verdict": str(parsed["impact_on_verdict"]),
                        "process_audit": str(parsed["process_audit"]),
                        "llm_fallback": False,
                        "llm_error": None,
                    }
                )
                continue

            # 工具级回退：单个工具总结失败不影响其他工具项。
            # 大白话：按“逐条容错”处理，避免一条坏数据毁掉整轮 Phase2 报告。
            summaries.append(
                {
                    "name": name,
                    "purpose": "LLM 分析失败，回退为规则占位。",
                    "input_reasonableness": f"参数: {json.dumps(arguments, ensure_ascii=False)}",
                    "result_core": (result[:180] + "...") if len(result) > 180 else (result or "(empty)"),
                    "impact_on_verdict": "回退模式：需人工判断该调用对裁决的真实影响。",
                    "process_audit": "回退模式：工具调用审查不完整。",
                    "llm_fallback": True,
                    "llm_error": err or "llm_invalid_payload",
                }
            )
        return summaries


class MarkdownRenderer:
    """Render structured markdown from collected events and summaries."""

    def __init__(self, summarizer: RoleSummarizer):
        self.summarizer = summarizer

    def render(self, run_data: dict) -> str:
        lines: list[str] = []
        # 先渲染全局头信息，再按 turn 顺序渲染阶段细节。
        # 大白话：输出顺序就是人类读报告的顺序，先看全貌，再看每轮细节。
        lines.append(f"# Aletheia 报告 - {run_data['problem_id']}")
        lines.append("")
        lines.append("### 元信息")
        lines.append(f"- **耗时**: {run_data['elapsed']:.1f}s")
        lines.append(f"- **迭代轮次**: {run_data['iteration_count']}")
        lines.append("")
        lines.append("### 问题描述与 Ground Truth")
        lines.append("```text")
        lines.append(run_data["problem_text"])
        lines.append("```")
        lines.append(f"- Ground Truth: `{run_data['ground_truth']}`")
        lines.append("")
        lines.append("### 最终生成结果")
        lines.append("```text")
        lines.append(str(run_data["final_output"]).strip())
        lines.append("```")
        lines.append("")
        lines.append("### 逐轮阶段追踪")

        turns: dict[int, list[dict]] = run_data["turns"]
        for turn_id in sorted(turns.keys()):
            turn_events = turns[turn_id]
            for event in turn_events:
                node = str(event.get("agent_node", ""))
                lines.append(f"#### Turn {turn_id} · {node}")

                if node in ("GENERATOR", "REVISER"):
                    # 生成/修订阶段：同时给“思维链摘要 + content摘要 + 原文”。
                    # 大白话：摘要用于快速浏览，原文用于可追责和细节复核。
                    reasoning = str(event.get("reasoning_content", "") or "")
                    reasoning_info = self.summarizer.summarize_reasoning(role=node, raw_cot=reasoning)
                    lines.append("- 思维链摘要：")
                    for idx, item in enumerate(reasoning_info.get("step_summary", []), start=1):
                        clean_item = self.summarizer.strip_leading_enumeration(str(item))
                        lines.append(f"  {idx}. {clean_item}")
                    lines.append("- 思维链质量评估：")
                    for item in reasoning_info.get("quality_evaluation", []):
                        lines.append(f"  - {item}")
                    if reasoning_info.get("llm_fallback"):
                        lines.append(f"- 思维链离线LLM失败原因：`{reasoning_info.get('llm_error')}`")

                    content = str(event.get("content", "") or "")
                    content_info = self.summarizer.summarize_role_content(node, content)
                    lines.append("- content 摘要：")
                    for item in content_info.get("content_summary", []):
                        lines.append(f"  - {item}")
                    lines.append("- content 质量评估：")
                    for item in content_info.get("content_quality", []):
                        lines.append(f"  - {item}")
                    if content_info.get("llm_fallback"):
                        lines.append(f"- content 离线LLM失败原因：`{content_info.get('llm_error')}`")

                    lines.append("- content 原文：")
                    lines.append("```text")
                    lines.append(content.strip())
                    lines.append("```")

                elif node == "VERIFIER":
                    # Verifier 按 Phase1/2/3 展开，保持和提示词协议一致，方便逐段审计。
                    phase1 = str(event.get("phase1_analysis", "") or "")
                    phase3 = str(event.get("full_verification_text", "") or "")
                    verification_report = str(event.get("verification_report", "") or "")
                    tool_calls = event.get("tool_calls_trace", []) or []

                    lines.append("- Phase 1（原文）：")
                    lines.append("```text")
                    lines.append(phase1.strip())
                    lines.append("```")

                    lines.append("- Phase 2（工具调用摘要与输入评估）：")
                    if tool_calls:
                        tool_summaries = self.summarizer.summarize_verifier_phase2_tools(tool_calls)
                        for idx, summary in enumerate(tool_summaries, start=1):
                            lines.append(f"  - [{idx}] `{summary['name']}`")
                            lines.append(f"    - 调用目的: {summary['purpose']}")
                            lines.append(f"    - 输入合理性: {summary['input_reasonableness']}")
                            lines.append(f"    - 返回核心结论: {summary['result_core']}")
                            lines.append(f"    - 对裁决作用: {summary['impact_on_verdict']}")
                            lines.append(f"    - 过程审查结论: {summary['process_audit']}")
                            if summary.get("llm_fallback"):
                                lines.append(f"    - LLM失败原因: `{summary.get('llm_error')}`")
                    else:
                        lines.append("  - (无工具调用)")

                    lines.append("- Phase 3（原文）：")
                    lines.append("```text")
                    lines.append(phase3.strip())
                    lines.append("```")

                    lines.append("- Verification Report（原文）：")
                    lines.append("```text")
                    lines.append(verification_report.strip())
                    lines.append("```")

                lines.append("")

        return "\n".join(lines)
