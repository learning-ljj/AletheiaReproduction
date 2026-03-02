"""IMOBench 全数据集批量评测实验脚本。

支持三类数据集，各取前 N 题（默认 30）逐题运行 Agent 并统计指标：

  answerbench  — 短答题：提取 \\boxed{} 答案与 ground truth 比对，统计 Exact Match Accuracy
  proofbench   — 证明题：统计证明格式完整率（has_preliminary_solution）和 CORRECT 裁决率
  gradingbench — 评分题：使用 Grader Agent 对已有 student response 打分，
                 与人工 Reward 标签（Correct/Partial/Almost/Incorrect）比对

用法：
    python scripts/run_imobench.py --dataset answerbench  [--count 30] [--max-turns 3]
    python scripts/run_imobench.py --dataset proofbench   [--count 30] [--max-turns 3]
    python scripts/run_imobench.py --dataset gradingbench [--count 30]
    python scripts/run_imobench.py --dataset all          [--count 30]

输出：
    控制台：逐题结果 + 汇总表格
    JSON  ：data/logs/imobench_{dataset}_{timestamp}.json

评分指标说明
-----------
answerbench:
  - exact_match_accuracy : 严格 LaTeX 归一化后与 ground truth 相同的比例

proofbench:
  - format_complete_rate : 解答含 "### Preliminary Solution ###" 标记的比例
  - correct_verdict_rate : 经 Verifier 裁决为 CORRECT 的比例（需 max_turns≥1）
  - has_final_answer_rate: state.final_answer 非 None 的比例

gradingbench:
  - grader_accuracy_binary: 二值准确率（Correct vs Non-Correct）
  - grader_accuracy_3way  : 三路准确率（Correct / Partial / Incorrect）
  - confusion_matrix      : 预测标签 vs 人工标签的混淆矩阵

  Grader 模式说明：
    gradingbench 中每条已有完整的学生回答（Response 字段），任务是判断该回答的质量。
    本脚本使用独立的 call_grader() 函数，构造专用 prompt 让 LLM 裁决，
    无需走完整的 Generator→Verifier 流水线。
    LLM 输出 [CORRECT] / [PARTIAL] / [INCORRECT] 标签，映射到人工 Reward 标签比较。
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

_W = 72


# ════════════════════════════════════════════════════════════════════════════
# Grader Agent：用于 gradingbench 的独立评分 LLM 调用
# ════════════════════════════════════════════════════════════════════════════

_GRADER_SYSTEM = """\
You are an expert IMO exam grader. Given a mathematical problem, its reference solution, \
the grading guidelines, and a student's response, you must assess the quality of the \
student's response.

Your judgment must be exactly one of:
  [CORRECT]   — The student's response is essentially complete and correct, \
with at most minor presentational issues.
  [PARTIAL]   — The student's response makes meaningful mathematical progress \
but is incomplete, contains gaps, or has non-critical errors.
  [INCORRECT] — The student's response is wrong, circular, or makes no significant progress.

Output your verdict tag on its own line first, then provide a brief explanation (2–4 sentences).\
"""

_GRADER_USER_TEMPLATE = """\
======================================================================
### Problem ###
{problem}

======================================================================
### Reference Solution ###
{solution}

======================================================================
### Grading Guidelines ###
{grading_guidelines}

======================================================================
### Student Response ###
{response}

======================================================================
### Grading Task ###
Assess the student response above. Output exactly one of [CORRECT], [PARTIAL], or [INCORRECT], \
followed by a brief explanation.\
"""

import re as _re
_GRADER_TAG_RE = _re.compile(r"\[(CORRECT|PARTIAL|INCORRECT)\]", _re.IGNORECASE)


def call_grader(
    llm_client,
    item: dict,
) -> tuple[str, str]:
    """对 gradingbench 的单条 item 调用 Grader LLM，返回 (tag, full_response)。

    tag 取值: "CORRECT" / "PARTIAL" / "INCORRECT" / "UNKNOWN"
    """
    user_content = _GRADER_USER_TEMPLATE.format(
        problem=item.get("problem", ""),
        solution=item.get("solution", ""),
        grading_guidelines=item.get("grading_guidelines", ""),
        response=item.get("response", ""),
    )
    messages = [
        {"role": "system", "content": _GRADER_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    resp = llm_client.chat(messages, thinking=False)
    text = resp.content or ""
    tags = _GRADER_TAG_RE.findall(text)
    tag = tags[0].upper() if tags else "UNKNOWN"
    return tag, text


def _human_to_3way(reward: str) -> str:
    """将人工 Reward 标签映射为三路标签。

    Correct → CORRECT
    Almost / Partial → PARTIAL
    Incorrect → INCORRECT
    """
    r = reward.strip().lower()
    if r == "correct":
        return "CORRECT"
    if r in ("partial", "almost"):
        return "PARTIAL"
    return "INCORRECT"


# ════════════════════════════════════════════════════════════════════════════
# AnswerBench 评测
# ════════════════════════════════════════════════════════════════════════════

def run_answerbench(agent, data: list[dict]) -> tuple[list[dict], dict]:
    """运行 AnswerBench，返回 (per-item results, summary)。"""
    from src.utils.evaluator import check_answer

    results = []
    for item in data:
        pid = item["problem_id"]
        t0 = time.time()
        try:
            state = agent.solve(pid, item["problem"])
            predicted = state.final_answer or state.current_proof
            correct = check_answer(predicted, item["answer"])
            elapsed = time.time() - t0
            # 采集 Agent 历史轮次信息（用于工作日志）
            history_info = [
                {
                    "turn_id": h.turn_id,
                    "agent_node": h.agent_node,
                    "decision": str(h.decision.value) if h.decision else None,
                    "tool_calls_count": len(h.tool_calls_trace) if h.tool_calls_trace else 0,
                    "bug_report_snippet": (h.bug_report or "")[:300] if h.bug_report else None,
                    "phase1_analysis": (h.phase1_analysis or "")[:500] if hasattr(h, "phase1_analysis") else None,
                }
                for h in state.history
            ]
            entry = {
                "problem_id": pid,
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
                "source": item.get("source", ""),
                "correct": correct,
                "iterations": state.iteration_count,
                "time_s": round(elapsed, 1),
                "ground_truth": item["answer"],
                "history": history_info,
            }
            status = "✅" if correct else "❌"
            print(f"  [{pid}] {status}  iters={state.iteration_count}  "
                  f"time={elapsed:.1f}s  gt={item['answer'][:30]}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
                "source": item.get("source", ""),
                "correct": False,
                "iterations": 0,
                "time_s": round(elapsed, 1),
                "error": str(exc),
                "history": [],
            }
            print(f"  [{pid}] ⚠️  Error: {exc}")
        results.append(entry)

    correct_count = sum(1 for r in results if r.get("correct"))
    total = len(results)
    summary = {
        "dataset": "answerbench",
        "total": total,
        "correct": correct_count,
        "exact_match_accuracy": round(correct_count / total, 4) if total else 0.0,
        "error_count": sum(1 for r in results if "error" in r),
    }
    return results, summary


# ════════════════════════════════════════════════════════════════════════════
# ProofBench 评测
# ════════════════════════════════════════════════════════════════════════════

def run_proofbench(agent, data: list[dict]) -> tuple[list[dict], dict]:
    """运行 ProofBench，返回 (per-item results, summary)。"""
    from src.utils.evaluator import check_proof_completeness
    from src.core.state import VerificationDecision

    results = []
    for item in data:
        pid = item["problem_id"]
        t0 = time.time()
        try:
            state = agent.solve(pid, item["problem"])
            predicted = state.final_answer or state.current_proof
            completeness = check_proof_completeness(predicted)
            # 最终 Verifier 裁决（取最后一个 VERIFIER 历史项）
            verifier_entries = [e for e in state.history if e.agent_node == "VERIFIER"]
            final_decision = (
                verifier_entries[-1].decision.value if verifier_entries else "NO_VERDICT"
            )
            elapsed = time.time() - t0
            # 采集 Agent 历史轮次信息（用于工作日志）
            history_info = [
                {
                    "turn_id": h.turn_id,
                    "agent_node": h.agent_node,
                    "decision": str(h.decision.value) if h.decision else None,
                    "tool_calls_count": len(h.tool_calls_trace) if h.tool_calls_trace else 0,
                    "bug_report_snippet": (h.bug_report or "")[:300] if h.bug_report else None,
                    "phase1_analysis": (h.phase1_analysis or "")[:500] if hasattr(h, "phase1_analysis") else None,
                }
                for h in state.history
            ]
            entry = {
                "problem_id": pid,
                "category": item.get("category", ""),
                "level": item.get("level", ""),
                "source": item.get("source", ""),
                "completeness": completeness,
                "final_verifier_decision": final_decision,
                "has_final_answer": state.final_answer is not None,
                "iterations": state.iteration_count,
                "time_s": round(elapsed, 1),
                "history": history_info,
            }
            status = "✅" if state.final_answer is not None else "⚠️"
            print(f"  [{pid}] {status}  decision={final_decision}  "
                  f"iters={state.iteration_count}  time={elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "category": item.get("category", ""),
                "level": item.get("level", ""),
                "source": item.get("source", ""),
                "completeness": check_proof_completeness(""),
                "final_verifier_decision": "ERROR",
                "has_final_answer": False,
                "iterations": 0,
                "time_s": round(elapsed, 1),
                "error": str(exc),
                "history": [],
            }
            print(f"  [{pid}] ⚠️  Error: {exc}")
        results.append(entry)

    total = len(results)
    format_ok = sum(
        1 for r in results
        if r.get("completeness", {}).get("has_preliminary_solution")
    )
    correct_v = sum(
        1 for r in results if r.get("final_verifier_decision") == "CORRECT"
    )
    has_ans = sum(1 for r in results if r.get("has_final_answer"))
    summary = {
        "dataset": "proofbench",
        "total": total,
        "format_complete": format_ok,
        "format_complete_rate": round(format_ok / total, 4) if total else 0.0,
        "correct_verdict": correct_v,
        "correct_verdict_rate": round(correct_v / total, 4) if total else 0.0,
        "has_final_answer": has_ans,
        "has_final_answer_rate": round(has_ans / total, 4) if total else 0.0,
        "error_count": sum(1 for r in results if "error" in r),
    }
    return results, summary


# ════════════════════════════════════════════════════════════════════════════
# GradingBench 评测
# ════════════════════════════════════════════════════════════════════════════

def run_gradingbench(llm_client, data: list[dict]) -> tuple[list[dict], dict]:
    """运行 GradingBench，对已有学生回答打分并与人工标签比对。

    注意：此函数不使用完整的 AletheiaAgent.solve()，而是直接调用 call_grader()，
    因为 gradingbench 的任务是评估一个给定的 response，而非从头生成解答。
    """
    results = []
    for item in data:
        pid = item["problem_id"]
        gid = item.get("grading_id", pid)
        t0 = time.time()
        try:
            pred_tag, grader_text = call_grader(llm_client, item)
            elapsed = time.time() - t0

            human_reward = item.get("reward", "").strip()
            human_3way = _human_to_3way(human_reward)
            human_binary = "CORRECT" if human_reward.lower() == "correct" else "INCORRECT"
            pred_binary = "CORRECT" if pred_tag == "CORRECT" else "INCORRECT"

            match_binary = pred_binary == human_binary
            match_3way = pred_tag == human_3way

            entry = {
                "grading_id": gid,
                "problem_id": pid,
                "predicted_tag": pred_tag,
                "human_reward": human_reward,
                "human_3way": human_3way,
                "match_binary": match_binary,
                "match_3way": match_3way,
                "human_points": item.get("points", 0),
                "time_s": round(elapsed, 1),
            }
            bm = "✅" if match_binary else "❌"
            tm = "✅" if match_3way else "△"
            print(f"  [{gid}] pred={pred_tag:12s} human={human_reward:10s}  "
                  f"binary={bm}  3way={tm}  time={elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "grading_id": gid,
                "problem_id": pid,
                "predicted_tag": "ERROR",
                "human_reward": item.get("reward", ""),
                "human_3way": _human_to_3way(item.get("reward", "")),
                "match_binary": False,
                "match_3way": False,
                "human_points": item.get("points", 0),
                "time_s": round(elapsed, 1),
                "error": str(exc),
            }
            print(f"  [{gid}] ⚠️  Error: {exc}")
        results.append(entry)

    total = len(results)
    binary_ok = sum(1 for r in results if r.get("match_binary"))
    way3_ok   = sum(1 for r in results if r.get("match_3way"))

    # 混淆矩阵（人工标签 → 预测标签）
    confusion: dict[str, Counter] = {
        "CORRECT":   Counter(),
        "PARTIAL":   Counter(),
        "INCORRECT": Counter(),
    }
    for r in results:
        confusion[r.get("human_3way", "INCORRECT")][r.get("predicted_tag", "UNKNOWN")] += 1

    summary = {
        "dataset": "gradingbench",
        "total": total,
        "grader_accuracy_binary": round(binary_ok / total, 4) if total else 0.0,
        "grader_accuracy_3way":   round(way3_ok   / total, 4) if total else 0.0,
        "binary_correct": binary_ok,
        "3way_correct": way3_ok,
        "error_count": sum(1 for r in results if "error" in r),
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
    return results, summary


# ════════════════════════════════════════════════════════════════════════════
# 输出汇总打印
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(summary: dict) -> None:
    dataset = summary["dataset"]
    print(f"\n{'═' * _W}")
    print(f"  Benchmark Summary: {dataset.upper()}")
    print(f"{'═' * _W}")
    for k, v in summary.items():
        if k == "confusion_matrix":
            print(f"  confusion_matrix (human→pred):")
            for human_label, pred_counts in v.items():
                print(f"    {human_label:12s}: {dict(pred_counts)}")
        else:
            print(f"  {k:<35s}: {v}")
    print(f"{'═' * _W}\n")


# ════════════════════════════════════════════════════════════════════════════
# 工作日志生成（Markdown）
# ════════════════════════════════════════════════════════════════════════════

def generate_worklog_markdown(
    all_results: dict[str, list[dict]],
    all_summaries: list[dict],
    start_time: datetime,
    output_path: str = "data/logs/imobench工作日志.md",
) -> None:
    """根据测试结果生成可读性强的 Markdown 工作日志。

    Args:
        all_results: {dataset_name: [per-item result dict, ...]}
        all_summaries: [summary dict, ...]
        start_time: 测试开始时间
        output_path: 输出 Markdown 路径
    """
    lines: list[str] = []
    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    def H(n: int, title: str) -> None:
        lines.append("#" * n + " " + title)
        lines.append("")

    def P(text: str) -> None:
        lines.append(text)
        lines.append("")

    def HR() -> None:
        lines.append("---")
        lines.append("")

    # ── 标题 ──────────────────────────────────────────────────────────
    H(1, "Aletheia IMOBench 评测工作日志")
    P(f"**测试时间**: {start_time.strftime('%Y-%m-%d %H:%M:%S')} — {end_time.strftime('%H:%M:%S')}"
      f"  （总耗时 {elapsed_total/60:.1f} 分钟）")
    P("**数据集**: AnswerBench (answerbench_v2) + ProofBench  |  **每题最大轮次**: 3")
    HR()

    # ── 汇总统计表 ────────────────────────────────────────────────────
    H(2, "汇总统计")
    for s in all_summaries:
        ds = s["dataset"]
        H(3, f"{ds.upper()} 汇总")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        for k, v in s.items():
            if k == "dataset":
                continue
            if k == "confusion_matrix":
                continue
            lines.append(f"| {k} | {v} |")
        lines.append("")

        if "confusion_matrix" in s:
            H(4, "混淆矩阵（人工标签 → 预测标签）")
            lines.append("| 人工标签 \\ 预测 | CORRECT | PARTIAL | INCORRECT | UNKNOWN |")
            lines.append("|-----------------|---------|---------|-----------|---------|")
            for human_label, pred_counts in s["confusion_matrix"].items():
                row_str = (
                    f"| **{human_label}** "
                    f"| {pred_counts.get('CORRECT',0)} "
                    f"| {pred_counts.get('PARTIAL',0)} "
                    f"| {pred_counts.get('INCORRECT',0)} "
                    f"| {pred_counts.get('UNKNOWN',0)} |"
                )
                lines.append(row_str)
            lines.append("")
    HR()

    # ── 逐题详细记录 ──────────────────────────────────────────────────
    for dataset, results in all_results.items():
        if not results:
            continue
        H(2, f"{dataset.upper()} 逐题过程记录")

        for idx, r in enumerate(results, 1):
            pid = r.get("problem_id", f"item_{idx}")
            cat = r.get("category", "")
            level = r.get("level", "")
            subcategory = r.get("subcategory", "")
            time_s = r.get("time_s", 0)
            iters = r.get("iterations", r.get("iteration_count", 0))
            error = r.get("error")

            # 题目标题行
            meta_parts = [x for x in [cat, subcategory, level] if x]
            meta_str = " · ".join(meta_parts) if meta_parts else ""
            H(3, f"[{idx}] {pid}" + (f"  `{meta_str}`" if meta_str else ""))

            # 基本信息
            lines.append(f"- **耗时**: {time_s:.1f}s  |  **迭代轮次**: {iters}")
            if dataset == "answerbench":
                correct = r.get("correct", False)
                gt = r.get("ground_truth", "")
                status = "✅ 正确" if correct else "❌ 错误"
                lines.append(f"- **结果**: {status}  |  **标准答案**: `{gt[:80]}`")
            elif dataset == "proofbench":
                decision = r.get("final_verifier_decision", "NO_VERDICT")
                has_ans = r.get("has_final_answer", False)
                fmt_ok = r.get("completeness", {}).get("has_preliminary_solution", False)
                lines.append(f"- **Verifier裁决**: `{decision}`  |  "
                              f"**格式完整**: {'✅' if fmt_ok else '❌'}  |  "
                              f"**有最终答案**: {'✅' if has_ans else '❌'}")
            lines.append("")

            # 错误记录
            if error:
                lines.append(f"> ⚠️ **运行错误**: `{error}`")
                lines.append("")

            # Agent 历史轮次记录
            history = r.get("history", [])
            if history:
                lines.append("**解题过程（Agent 历史）**:")
                lines.append("")
                lines.append("| 轮次 | Agent | 裁决 | 工具调用 | Bug/警告摘要 |")
                lines.append("|------|-------|------|----------|--------------|")
                for h in history:
                    turn_id = h.get("turn_id", "?")
                    node = h.get("agent_node", "?")
                    dec = h.get("decision") or "—"
                    tc = h.get("tool_calls_count", 0)
                    bug_snip = (h.get("bug_report_snippet") or "—")[:80].replace("\n", " ")
                    lines.append(f"| {turn_id} | {node} | `{dec}` | {tc} | {bug_snip} |")
                lines.append("")

                # 详细 Phase1 分析（仅 VERIFIER）
                for h in history:
                    if h.get("agent_node") == "VERIFIER" and h.get("phase1_analysis"):
                        p1 = (h.get("phase1_analysis") or "")[:500].strip()
                        if p1:
                            lines.append(f"<details><summary>Turn {h.get('turn_id')} Verifier Phase1 分析摘要</summary>")
                            lines.append("")
                            lines.append("```")
                            lines.append(p1)
                            lines.append("```")
                            lines.append("")
                            lines.append("</details>")
                            lines.append("")

                # Bug 详情（CRITICAL_FLAW / MINOR_FLAW）
                for h in history:
                    bug = h.get("bug_report_snippet")
                    if bug and h.get("decision") in ("CRITICAL_FLAW", "MINOR_FLAW"):
                        lines.append(f"**Turn {h.get('turn_id')} Bug 报告摘要** (`{h.get('decision')}`):")
                        lines.append("")
                        lines.append("> " + bug[:300].replace("\n", "\n> "))
                        lines.append("")

            lines.append("---")
            lines.append("")

    # ── Bug & 警告汇总 ────────────────────────────────────────────────
    H(2, "Bug 与警告汇总")
    bug_count = 0
    for dataset, results in all_results.items():
        for r in results:
            if r.get("error"):
                bug_count += 1
                pid = r.get("problem_id", "?")
                lines.append(f"- **[{dataset}] {pid}**: {r['error'][:200]}")
    if bug_count == 0:
        P("本次测试未发现运行时错误。")
    else:
        P(f"共发现 {bug_count} 道题目运行出错，详见上方逐题记录。")
    HR()

    H(2, "复盘分析说明")
    lines.append("本日志由 `run_imobench.py` 自动生成，记录了 Aletheia 框架在 IMOBench 上的完整评测过程。")
    lines.append("")
    lines.append("**分析要点**：")
    lines.append("- 关注 AnswerBench 中 Verifier 裁决为 CRITICAL_FLAW 的题目，分析是否是 Generator 解题方向有误。")
    lines.append("- 关注 ProofBench 中格式不完整（无 `### Preliminary Solution ###` 标记）的情况。")
    lines.append("- 工具调用次数为 0 但仍裁决为 CRITICAL_FLAW 的情况，可能说明 Verifier 仅靠阅读分析就发现了问题。")
    lines.append("- 超时（time_s > 300）说明某一环节（通常是 Phase 2 工具调用）耗时过长，需关注网络稳定性。")
    lines.append("")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 工作日志已保存至: {out}\n")


# ════════════════════════════════════════════════════════════════════════════
# 数据加载（带 gradingbench 的 grading_guidelines 字段修正）
# ════════════════════════════════════════════════════════════════════════════

def load_gradingbench_full(path: str = "data/imobench/gradingbench.csv") -> list[dict]:
    """加载 gradingbench 并保留 Grading guidelines 字段（data_loader 默认不加载此列）。"""
    import csv
    from pathlib import Path as _Path
    results = []
    with open(_Path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            results.append({
                "problem_id": f"gradingbench_{i:04d}",
                "grading_id": row.get("Grading ID", f"GB-{i:04d}"),
                "source_problem_id": row.get("Problem ID", ""),
                "problem": row.get("Problem", ""),
                "solution": row.get("Solution", ""),
                "grading_guidelines": row.get("Grading guidelines", ""),
                "response": row.get("Response", ""),
                "points": _safe_int(row.get("Points", "0")),
                "reward": row.get("Reward", ""),
                "problem_source": row.get("Problem Source", ""),
            })
    return results


def load_proofbench_full(path: str = "data/imobench/proofbench.csv") -> list[dict]:
    """加载 proofbench 完整字段（含 Category、Level、Source）。"""
    import csv
    from pathlib import Path as _Path
    results = []
    with open(_Path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            results.append({
                "problem_id": row.get("Problem ID", f"proofbench_{i:04d}"),
                "problem": row.get("Problem", ""),
                "solution": row.get("Solution", ""),
                "category": row.get("Category", ""),
                "level": row.get("Level", ""),
                "source": row.get("Source", ""),
            })
    return results


def load_answerbench_full(path: str = "data/imobench/answerbench_v2.csv") -> list[dict]:
    """加载 answerbench_v2 完整字段（含 Category、Subcategory、Source）。"""
    import csv
    from pathlib import Path as _Path
    results = []
    with open(_Path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            results.append({
                "problem_id": row.get("Problem ID", f"answerbench_{i:04d}"),
                "problem": row.get("Problem", ""),
                "answer": row.get("Short Answer", ""),
                "category": row.get("Category", ""),
                "subcategory": row.get("Subcategory", ""),
                "source": row.get("Source", ""),
            })
    return results


def select_proofbench_diverse(data: list[dict], n: int = 10) -> list[dict]:
    """从 proofbench 全集按类别+难度多样性选取 n 道题。

    策略：优先覆盖不同类别（Algebra/Combinatorics/Number theory/Geometry）和
    不同难度层次（pre-IMO → IMO-easy → IMO-medium → IMO-hard），
    用轮询方式均衡抽取。
    """
    from collections import defaultdict
    # 按 (category, level) 分组
    groups: dict[tuple, list] = defaultdict(list)
    for item in data:
        key = (item.get("category", ""), item.get("level", ""))
        groups[key].append(item)

    categories = ["Algebra", "Combinatorics", "Number theory", "Geometry"]
    levels = ["pre-IMO", "IMO-easy", "IMO-medium", "IMO-hard"]

    selected: list[dict] = []
    # 第一轮：按难度层次，逐类别取一道
    for level in levels:
        for cat in categories:
            if len(selected) >= n:
                break
            key = (cat, level)
            if groups[key]:
                selected.append(groups[key].pop(0))
        if len(selected) >= n:
            break

    # 补充：若仍不足 n 道，从剩余中按原始顺序补
    remaining = [item for items in groups.values() for item in items]
    for item in remaining:
        if len(selected) >= n:
            break
        selected.append(item)

    return selected[:n]


def select_answerbench_diverse(data: list[dict], n: int = 30) -> list[dict]:
    """从 answerbench_v2 全集按类别+子类别多样性选取 n 道题。

    策略：均衡地从 Algebra、Combinatorics、Geometry、Number theory 四类中抽取，
    同一类别内按 Subcategory 去重以保证子类别多样性。
    """
    from collections import defaultdict
    main_cats = ["Algebra", "Combinatorics", "Geometry", "Number theory"]

    # 按 category 分组，并在类别内按 subcategory 轮询
    cat_groups: dict[str, list] = defaultdict(list)
    other: list[dict] = []
    for item in data:
        cat = item.get("category", "")
        if cat in main_cats:
            cat_groups[cat].append(item)
        else:
            other.append(item)

    # 每类分配额度（尽量均等）
    per_cat = n // len(main_cats)  # 7
    extra   = n % len(main_cats)   # 2

    selected: list[dict] = []
    for i, cat in enumerate(main_cats):
        quota = per_cat + (1 if i < extra else 0)
        items = cat_groups[cat]
        # 在类别内按 subcategory 轮询以保证子类别多样性
        sub_groups: dict[str, list] = defaultdict(list)
        for item in items:
            sub_groups[item.get("subcategory", "")].append(item)
        sub_queues = list(sub_groups.values())
        taken = 0
        while taken < quota:
            progress = False
            for q in sub_queues:
                if q and taken < quota:
                    selected.append(q.pop(0))
                    taken += 1
                    progress = True
            if not progress:
                break

    # 如果因类别不均等导致不足，从 other 补充
    for item in other:
        if len(selected) >= n:
            break
        selected.append(item)

    return selected[:n]


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


# ════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMOBench 全数据集实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["answerbench", "proofbench", "gradingbench", "all"],
        help="要评测的数据集（all = 三类全部运行）",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="每个数据集取前 N 题（answerbench 默认30，proofbench 默认10，gradingbench 默认30）；"
             "0 表示取全部",
    )
    parser.add_argument(
        "--max-turns", type=int, default=3,
        help="answerbench/proofbench 每题最大 Agent 轮次（默认 3）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/logs",
        help="结果 JSON 输出目录",
    )
    parser.add_argument(
        "--no-worklog", action="store_true",
        help="禁用 Markdown 工作日志生成",
    )
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────
    from src.core.config import load_config, load_prompts
    from src.core.agent import AletheiaAgent
    from src.models.llm_client import create_llm_client

    config = load_config()
    prompts = load_prompts()
    config.setdefault("agent", {})["max_turns"] = args.max_turns

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = (
        ["answerbench", "proofbench", "gradingbench"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_summaries: list[dict] = []
    all_results: dict[str, list[dict]] = {}

    for dataset in datasets_to_run:
        # ── 数据集默认题数 ──────────────────────────────────────────────
        default_counts = {"answerbench": 30, "proofbench": 10, "gradingbench": 30}
        count = args.count if args.count is not None else default_counts.get(dataset, 30)

        print(f"\n{'═' * _W}")
        print(f"  Dataset: {dataset.upper()}  |  Count: {count}  |  "
              f"Max turns: {args.max_turns}")
        print(f"{'═' * _W}\n")

        if dataset == "answerbench":
            # 从 answerbench_v2 加载全字段，按类别多样性选题
            full_data = load_answerbench_full()
            if count == 0:
                data = full_data
            else:
                data = select_answerbench_diverse(full_data, n=count)
            agent = AletheiaAgent(config, prompts)
            results, summary = run_answerbench(agent, data)

        elif dataset == "proofbench":
            # 从 proofbench 加载全字段，按类别+难度多样性选题
            full_data = load_proofbench_full()
            if count == 0:
                data = full_data
            else:
                data = select_proofbench_diverse(full_data, n=count)
            agent = AletheiaAgent(config, prompts)
            results, summary = run_proofbench(agent, data)

        else:  # gradingbench
            data = load_gradingbench_full()
            if count > 0:
                data = data[:count]
            llm_client = create_llm_client(config)
            results, summary = run_gradingbench(llm_client, data)

        _print_summary(summary)
        all_summaries.append(summary)
        all_results[dataset] = results

        # 保存单数据集结果
        out_path = output_dir / f"imobench_{dataset}_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Results saved to: {out_path}\n")

    # 若运行了多个数据集，打印总汇总
    if len(all_summaries) > 1:
        print(f"\n{'═' * _W}")
        print("  OVERALL SUMMARY")
        print(f"{'═' * _W}")
        for s in all_summaries:
            ds = s["dataset"]
            if ds == "answerbench":
                print(f"  answerbench  exact_match_accuracy = {s['exact_match_accuracy']:.2%}  "
                      f"({s['correct']}/{s['total']})")
            elif ds == "proofbench":
                print(f"  proofbench   correct_verdict_rate = {s['correct_verdict_rate']:.2%}  "
                      f"({s['correct_verdict']}/{s['total']})")
            else:
                print(f"  gradingbench grader_accuracy_3way  = {s['grader_accuracy_3way']:.2%}  "
                      f"({s['3way_correct']}/{s['total']})")
        print(f"{'═' * _W}\n")

    # ── 生成 Markdown 工作日志 ─────────────────────────────────────────
    if not args.no_worklog and ("answerbench" in all_results or "proofbench" in all_results):
        worklog_path = output_dir / f"imobench工作日志_{timestamp}.md"
        generate_worklog_markdown(
            all_results={k: v for k, v in all_results.items() if k in ("answerbench", "proofbench")},
            all_summaries=[s for s in all_summaries if s["dataset"] in ("answerbench", "proofbench")],
            start_time=start_time,
            output_path=str(worklog_path),
        )


if __name__ == "__main__":
    main()
