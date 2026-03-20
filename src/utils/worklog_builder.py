"""工作日志构建器：从 raw JSONL 离线生成结构化 Markdown。"""

import json
import threading
from datetime import datetime
from pathlib import Path

from src.models.llm_client import create_llm_client
from src.utils.raw_log_reader import load_raw_events


class WorklogBuilder:
    """工作日志生成器。"""

    def __init__(self, llm_client=None, llm_config: dict | None = None):
        self.llm_client = llm_client
        self.llm_config = llm_config
        self._active_llm_client = None
        self._active_llm_error = None
        self._llm_timeout_seconds = self._resolve_llm_timeout_seconds(llm_config)

    @staticmethod
    def _resolve_llm_timeout_seconds(llm_config: dict | None) -> float:
        """读取 Worklog 摘要调用的硬超时（秒）。"""
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

    def _begin_worklog_session(self) -> None:
        """为当前 worklog 准备专用 LLM client。"""
        self._active_llm_error = None
        if self.llm_config is not None:
            try:
                # 使用默认 stream 行为：在终端/重定向日志中可实时观察 Worklog 摘要生成进度。
                self._active_llm_client = create_llm_client(self.llm_config)
            except Exception as exc:  # noqa: BLE001
                self._active_llm_client = None
                self._active_llm_error = f"llm_client_init_exception:{type(exc).__name__}:{exc}"
            return
        self._active_llm_client = self.llm_client

    def _end_worklog_session(self) -> None:
        self._active_llm_client = None
        self._active_llm_error = None

    @staticmethod
    def _clip(text: str, max_chars: int = 3200) -> str:
        s = (text or "").strip()
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + f"\n...[truncated {len(s) - max_chars} chars]"

    @staticmethod
    def _extract_json(text: str) -> dict:
        s = (text or "").strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                s = "\n".join(lines[1:-1]).strip()
        return json.loads(s)

    @staticmethod
    def _as_text_list(value) -> list[str] | None:
        if isinstance(value, list):
            return [str(x) for x in value]
        if isinstance(value, str):
            return [value]
        return None

    def _llm_json(self, system: str, user: str, required_keys: list[str]) -> tuple[dict | None, str | None]:
        client = self._active_llm_client or self.llm_client
        if client is None:
            if self._active_llm_error:
                return None, self._active_llm_error
            return None, "llm_client_not_configured"

        response_holder: dict[str, object] = {}

        def _call_llm() -> None:
            try:
                response_holder["resp"] = client.chat(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    thinking=False,
                    stream_prefix="WORKLOG",
                )
            except Exception as exc:  # noqa: BLE001
                response_holder["exc"] = exc

        worker = threading.Thread(target=_call_llm, daemon=True)
        worker.start()
        worker.join(timeout=self._llm_timeout_seconds)
        if worker.is_alive():
            return None, f"llm_timeout:{self._llm_timeout_seconds:.1f}s"

        if "exc" in response_holder:
            exc = response_holder["exc"]
            return None, f"llm_exception:{type(exc).__name__}:{exc}"

        try:
            resp = response_holder.get("resp")
            if resp is None:
                return None, "llm_missing_response"
            parsed = self._extract_json(resp.content or "")
            for k in required_keys:
                if k not in parsed:
                    return None, f"llm_missing_key:{k}"
            return parsed, None
        except Exception as exc:  # noqa: BLE001
            return None, f"llm_exception:{type(exc).__name__}:{exc}"

    def _summarize_reasoning(self, role: str, raw_cot: str) -> dict:
        cot = (raw_cot or "").strip()
        if not cot:
            return {
                "step_summary": ["(empty)"],
                "quality_evaluation": ["思维链为空，无法评估。"],
                "llm_fallback": True,
                "llm_error": "empty_reasoning",
            }

        prompt = (
            "你是数学解题过程审阅器。仅输出合法 JSON，不要输出其它文本。\n"
            "JSON 键必须是：step_summary, quality_evaluation。\n"
            "要求：\n"
            "1) step_summary：按原思考顺序分步骤概括关键推导、关键假设、计算要点、结论。\n"
            "2) quality_evaluation：逐步指出冗余、明显错误、推理断裂、未经证实假设、与题意冲突点；"
            "若无明显问题也要明确写出。\n"
            f"角色: {role}\n"
            f"思维链原文:\n{self._clip(cot)}"
        )
        parsed, err = self._llm_json(
            system="你只输出 JSON。",
            user=prompt,
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

        prompt = (
            "你是数学解答质量评估器。仅输出合法 JSON，不要输出其它文本。\n"
            "JSON 键必须是：content_summary, content_quality。\n"
            "要求：\n"
            "1) content_summary：概括解答结构、关键步骤、最终结论。\n"
            "2) content_quality：评估答案完整性、与思维链一致性、是否有明显算术/逻辑错误、"
            "是否缺少关键前提或结论。\n"
            f"角色: {role}\n"
            f"content 原文:\n{self._clip(text)}"
        )
        parsed, err = self._llm_json(
            system="你只输出 JSON。",
            user=prompt,
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

            prompt = (
                "你是数学验证过程审阅器。仅输出合法 JSON，不要输出其它文本。\n"
                "JSON 键必须是：purpose, input_reasonableness, result_core, impact_on_verdict, process_audit。\n"
                "要求：\n"
                "1) purpose：调用目的（验证什么、为什么要调）。\n"
                "2) input_reasonableness：输入参数与简短说明，并分析输入是否合理。\n"
                "3) result_core：返回结果的核心结论（禁止粘贴长文）。\n"
                "4) impact_on_verdict：该调用对裁决/结论的作用。\n"
                "5) process_audit：问题验证/过程审查结论。\n"
                f"tool_name: {name}\n"
                f"tool_arguments: {json.dumps(arguments, ensure_ascii=False)}\n"
                f"tool_result: {self._clip(result, max_chars=1800)}"
            )
            parsed, err = self._llm_json(
                system="你只输出 JSON。",
                user=prompt,
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

            # 失败回退：保底给出结构化信息，并显式记录失败原因。
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

    @staticmethod
    def _parse_ts(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    def build_problem_worklog(self, run_jsonl_path: str, output_md_path: str) -> None:
        """读取 raw 日志并生成最终规范的工作日志 Markdown。"""
        self._begin_worklog_session()
        try:
            run_path = Path(run_jsonl_path)
            output_path = Path(output_md_path)

            problem_id = run_path.stem
            events = load_raw_events(problem_id=problem_id, log_dir=run_path.parent)

            # 元信息：耗时与迭代轮次从事件序列推导。
            timestamps = [self._parse_ts(e.get("timestamp")) for e in events]
            timestamps = [t for t in timestamps if t is not None]
            elapsed = 0.0
            if len(timestamps) >= 2:
                elapsed = (max(timestamps) - min(timestamps)).total_seconds()

            tracked_nodes = {"GENERATOR", "VERIFIER", "REVISER"}
            turn_ids = [
                int(e.get("turn_id", 0))
                for e in events
                if str(e.get("agent_node")) in tracked_nodes
            ]
            iteration_count = max(turn_ids) if turn_ids else 0

            # 取问题与 GT：从首条事件元信息中读取。
            first_event = events[0] if events else {}
            problem_text = first_event.get("problem_text") or first_event.get("input", {}).get("problem_text") or ""
            ground_truth = first_event.get("ground_truth") or ""

            final_event = next((e for e in reversed(events) if e.get("agent_node") == "FINAL"), {})
            final_output = final_event.get("final_output") or final_event.get("content") or ""

            # 按 turn 组织事件，便于模板按轮渲染。
            turns: dict[int, list[dict]] = {}
            for event in events:
                node = str(event.get("agent_node", ""))
                if node not in tracked_nodes:
                    continue
                turn_id = int(event.get("turn_id", 0))
                turns.setdefault(turn_id, []).append(event)

            lines: list[str] = []
            lines.append(f"# Aletheia 报告 - {problem_id}")
            lines.append("")
            lines.append("### 元信息")
            lines.append(f"- **耗时**: {elapsed:.1f}s")
            lines.append(f"- **迭代轮次**: {iteration_count}")
            lines.append("")
            lines.append("### 问题描述与 Ground Truth")
            lines.append("```text")
            lines.append(problem_text)
            lines.append("```")
            lines.append(f"- Ground Truth: `{ground_truth}`")
            lines.append("")
            lines.append("### 最终生成结果")
            lines.append("```text")
            lines.append(str(final_output).strip())
            lines.append("```")
            lines.append("")
            lines.append("### 逐轮阶段追踪")

            for turn_id in sorted(turns.keys()):
                turn_events = turns[turn_id]

                for event in turn_events:
                    node = str(event.get("agent_node", ""))
                    lines.append(f"#### Turn {turn_id} · {node}")

                    if node in ("GENERATOR", "REVISER"):
                        # 思维链只展示摘要与评估，不展示原文。
                        reasoning = str(event.get("reasoning_content", "") or "")
                        reasoning_info = self._summarize_reasoning(role=node, raw_cot=reasoning)
                        lines.append("- 思维链摘要：")
                        for idx, item in enumerate(reasoning_info.get("step_summary", []), start=1):
                            lines.append(f"  {idx}. {item}")
                        lines.append("- 思维链质量评估：")
                        for item in reasoning_info.get("quality_evaluation", []):
                            lines.append(f"  - {item}")
                        if reasoning_info.get("llm_fallback"):
                            lines.append(f"- 思维链离线LLM失败原因：`{reasoning_info.get('llm_error')}`")

                        # content 做摘要与质量评估，但原文必须完整展示。
                        content = str(event.get("content", "") or "")
                        content_info = self.summarize_role_content(node, content)
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
                        # Verifier Phase1/Phase3/Verification Report 按要求展示原文。
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
                            tool_summaries = self.summarize_verifier_phase2_tools(tool_calls)
                            for i, node_sum in enumerate(tool_summaries, start=1):
                                lines.append(f"  - [{i}] `{node_sum['name']}`")
                                lines.append(f"    - 调用目的: {node_sum['purpose']}")
                                lines.append(f"    - 输入合理性: {node_sum['input_reasonableness']}")
                                lines.append(f"    - 返回核心结论: {node_sum['result_core']}")
                                lines.append(f"    - 对裁决作用: {node_sum['impact_on_verdict']}")
                                lines.append(f"    - 过程审查结论: {node_sum['process_audit']}")
                                if node_sum.get("llm_fallback"):
                                    lines.append(f"    - LLM失败原因: `{node_sum.get('llm_error')}`")
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

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
        finally:
            self._end_worklog_session()
