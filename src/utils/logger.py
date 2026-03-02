"""JSONL 日志持久化与格式化控制台/文件输出。"""

import json
from pathlib import Path

from src.core.state import VerificationLog

# 默认日志目录
LOG_DIR = Path("data/logs")


def append_log(problem_id: str, log_entry: VerificationLog, log_dir: Path = LOG_DIR) -> None:
    """追加一条日志到 {log_dir}/{problem_id}.jsonl"""
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{problem_id}.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(log_entry.model_dump_json() + "\n")


def read_logs(problem_id: str, log_dir: Path = LOG_DIR) -> list[VerificationLog]:
    """读取指定 problem 的全部日志。"""
    filepath = log_dir / f"{problem_id}.jsonl"
    if not filepath.exists():
        return []
    logs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(VerificationLog.model_validate_json(line))
    return logs


# ─── 控制台格式化输出 ─────────────────────────────────────────────────────────

_W = 72  # 控制台输出宽度

_AGENT_ICON: dict[str, str] = {
    "GENERATOR": "🔧",
    "VERIFIER":  "🔬",
    "REVISER":   "✏️ ",
}

_VERDICT_LABEL: dict[str, str] = {
    "CORRECT":       "✅  [CORRECT]        — solution accepted",
    "MINOR_FLAW":    "⚠️   [MINOR_FLAW]     — justification gap(s) found",
    "CRITICAL_FLAW": "❌  [CRITICAL_FLAW]  — critical error(s) found",
}


def _hr(char: str = "─", width: int = _W) -> str:
    return char * width


def _trunc(text: str, max_chars: int = 600) -> str:
    """截断长文本并附加省略说明。"""
    if not text or not text.strip():
        return "(empty)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n  … [{omitted} chars omitted / {len(text)} total]"


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def print_log_entry(entry: VerificationLog) -> None:
    """将单条 VerificationLog 以结构化格式打印到 stdout，按 agent 类型分区显示。"""
    node = entry.agent_node
    icon = _AGENT_ICON.get(node, "❓")
    ts = entry.timestamp[:19].replace("T", " ") if entry.timestamp else ""

    # ── 标题行 ───────────────────────────────────────────────────────────────
    print(f"\n{_hr('━')}")
    title = f"{icon} [{node}]  Turn {entry.turn_id}"
    print(f"{title:<56}{ts:>16}")
    print(_hr("━"))

    # ── Generator / Reviser ──────────────────────────────────────────────────
    if node in ("GENERATOR", "REVISER"):
        cot = (entry.extracted_cot or "").strip()
        if cot:
            print(f"\n🧠 Chain-of-Thought  ({len(cot):,} chars)")
            print(_hr())
            print(_indent(_trunc(cot, 500)))

        content = (entry.content or "").strip()
        print(f"\n📝 Solution Output  ({len(content):,} chars)")
        print(_hr())
        print(_indent(_trunc(content, 800)))

    # ── Verifier ─────────────────────────────────────────────────────────────
    elif node == "VERIFIER":
        # Phase 1 初步分析
        p1 = (entry.phase1_analysis or "").strip()
        if p1:
            print(f"\n📋 Phase 1 — Initial Analysis  ({len(p1):,} chars)")
            print(_hr())
            print(_indent(_trunc(p1, 500)))

        # Phase 2 工具调用
        traces = entry.tool_calls_trace or []
        print(f"\n🔬 Phase 2 — Tool Calls  ({len(traces)} call(s))")
        print(_hr())
        if traces:
            for i, tc in enumerate(traces, 1):
                name = tc.get("name", "?")
                args = tc.get("arguments", {})
                result = str(tc.get("result", ""))
                print(f"\n  [{i}] {name}")
                for k, v in args.items():
                    v_str = str(v).replace("\n", "↵ ")
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "…"
                    print(f"      ▸ {k}: {v_str}")
                print("      ◀ Result:")
                print(_indent(_trunc(result, 400), 8))
        else:
            print("  (no tool calls)")

        # Phase 3 裁决
        d = entry.decision
        verdict_label = _VERDICT_LABEL.get(d.value if d else "", f"[{d}]")
        print(f"\n{verdict_label}")
        print(_hr())

        bug = (entry.bug_report or "").strip()
        if bug:
            print(f"\n📌 Bug Report  ({len(bug):,} chars)")
            print(_hr())
            print(_indent(_trunc(bug, 600)))

        full = (entry.full_verification_text or "").strip()
        if full:
            print(f"\n📄 Phase 3 — Full Verdict  ({len(full):,} chars)")
            print(_hr())
            print(_indent(_trunc(full, 600)))

    print(f"\n{_hr('─')}\n")


# ─── 文件格式化输出 ───────────────────────────────────────────────────────────

def write_readable_log(
    problem_id: str,
    problem_text: str,
    history: list[VerificationLog],
    final_answer: str | None,
    filepath: Path,
) -> None:
    """将完整任务结果写入格式化的人类可读日志文件。"""
    W = 72
    SEP = "=" * W
    sep = "-" * W

    lines: list[str] = []
    verifier_turns = [e for e in history if e.agent_node == "VERIFIER"]

    lines.append(SEP)
    lines.append("  ALETHEIA RUN LOG")
    lines.append(SEP)
    lines.append(f"  Problem ID   : {problem_id}")
    lines.append(f"  Iterations   : {len(verifier_turns)}")
    lines.append(f"  Final Answer : {'FOUND' if final_answer else 'NOT FOUND (max turns exhausted)'}")
    lines.append(SEP)
    lines.append("")

    for entry in history:
        node = entry.agent_node
        ts = entry.timestamp[:19].replace("T", " ") if entry.timestamp else ""
        lines.append(sep)
        lines.append(f"  [{node}]  Turn {entry.turn_id}  |  {ts}")
        lines.append(sep)
        lines.append("")

        if node in ("GENERATOR", "REVISER"):
            cot = (entry.extracted_cot or "").strip()
            if cot:
                lines.append(f"  [THINKING]  ({len(cot):,} chars)")
                lines.append("")
                for line in cot[:1500].splitlines():
                    lines.append("    " + line)
                if len(cot) > 1500:
                    lines.append(f"    … [{len(cot) - 1500} chars omitted]")
                lines.append("")

            content = entry.content or ""
            lines.append(f"  [SOLUTION OUTPUT]  ({len(content):,} chars)")
            lines.append("")
            for line in content.splitlines():
                lines.append("    " + line)

        elif node == "VERIFIER":
            p1 = (entry.phase1_analysis or "").strip()
            if p1:
                lines.append(f"  [PHASE 1 — INITIAL ANALYSIS]  ({len(p1):,} chars)")
                lines.append("")
                for line in p1[:1000].splitlines():
                    lines.append("    " + line)
                if len(p1) > 1000:
                    lines.append(f"    … [{len(p1) - 1000} chars omitted]")
                lines.append("")

            traces = entry.tool_calls_trace or []
            lines.append(f"  [PHASE 2 — TOOL CALLS]  ({len(traces)} call(s))")
            lines.append("")
            if traces:
                for i, tc in enumerate(traces, 1):
                    lines.append(f"    [{i}] Tool: {tc.get('name')}")
                    args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                    if len(args_str) > 300:
                        args_str = args_str[:300] + "…"
                    lines.append(f"        Args  : {args_str}")
                    result_str = str(tc.get("result", ""))
                    if len(result_str) > 500:
                        result_str = result_str[:500] + f"… [{len(result_str)} total chars]"
                    lines.append("        Result:")
                    for line in result_str.splitlines():
                        lines.append("          " + line)
                    lines.append("")
            else:
                lines.append("    (no tool calls)")
                lines.append("")

            d = entry.decision
            d_val = d.value if d else "N/A"
            lines.append(f"  [PHASE 3 — VERDICT]  {d_val}")
            lines.append("")

            bug = (entry.bug_report or "").strip()
            if bug:
                lines.append(f"  [BUG REPORT]  ({len(bug):,} chars)")
                lines.append("")
                for line in bug[:1500].splitlines():
                    lines.append("    " + line)
                if len(bug) > 1500:
                    lines.append(f"    … [{len(bug) - 1500} chars omitted]")
                lines.append("")

            full = (entry.full_verification_text or "").strip()
            if full:
                lines.append(f"  [FULL VERDICT TEXT]  ({len(full):,} chars)")
                lines.append("")
                for line in full[:2000].splitlines():
                    lines.append("    " + line)
                if len(full) > 2000:
                    lines.append(f"    … [{len(full) - 2000} chars omitted]")

        lines.append("")
        lines.append("")

    if final_answer:
        lines.append(SEP)
        lines.append("  FINAL ANSWER")
        lines.append(SEP)
        lines.append("")
        for line in final_answer.splitlines():
            lines.append("  " + line)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines), encoding="utf-8")
