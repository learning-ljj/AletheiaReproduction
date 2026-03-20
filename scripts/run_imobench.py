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


def _build_run_id(problem_id: str) -> str:
    """为单题运行构造唯一日志键，避免历史 JSONL 混写。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{problem_id}_{ts}"


def _serialize_history(history) -> list[dict]:
    """将 Agent history 序列化为 raw-only 结构。"""
    output: list[dict] = []
    for h in history:
        node = getattr(h, "agent_node", "")
        entry: dict = {
            "turn_id": getattr(h, "turn_id", None),
            "agent_node": node,
            "decision": str(h.decision.value) if getattr(h, "decision", None) else None,
            "tool_calls_count": len(getattr(h, "tool_calls_trace", []) or []),
            "timestamp": getattr(h, "timestamp", None),
        }

        if node in ("GENERATOR", "REVISER"):
            cot = getattr(h, "extracted_cot", "") or ""
            content = getattr(h, "content", "") or ""
            entry.update({
                "reasoning_full": cot,
                "content_full": content,
            })
        elif node == "VERIFIER":
            phase1 = getattr(h, "phase1_analysis", "") or ""
            verification_report = getattr(h, "verification_report", "") or ""
            full_verification = getattr(h, "full_verification_text", "") or ""
            tool_calls = getattr(h, "tool_calls_trace", []) or []
            entry.update({
                "phase1_full": phase1,
                "tool_calls": tool_calls,
                "phase3_full": full_verification,
                "verification_report": verification_report,
                "parse_error": getattr(h, "parse_error", None),
            })
        output.append(entry)
    return output


def _last_verifier_decision(history: list[dict]) -> str | None:
    verifier = [x for x in history if x.get(“agent_node”) == “VERIFIER”]
    return verifier[-1].get(“decision”) if verifier else None


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
    from src.utils.parser import extract_xml_tag, normalize_short_answer

    results = []
    for item in data:
        pid = item["problem_id"]
        run_id = _build_run_id(pid)
        t0 = time.time()
        try:
            state = agent.solve(run_id, item["problem"], ground_truth=item.get("answer", ""))
            predicted = state.final_answer or state.current_proof
            # 如果 Generator/Verifier 使用了 XML 输出（<verdict>），优先从中提取短答并归一化
            verdict_candidate = extract_xml_tag(predicted, "verdict")
            if verdict_candidate:
                predicted_for_check = normalize_short_answer(verdict_candidate)
            else:
                # 也对直接的预测文本做归一化，减少格式差异带来的不匹配
                predicted_for_check = normalize_short_answer(predicted)

            correct = check_answer(predicted_for_check, item["answer"])
            elapsed = time.time() - t0
            history_info = _serialize_history(state.history)
            final_verifier_decision = _last_verifier_decision(history_info)
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
                "source": item.get("source", ""),
                "correct": correct,
                "iterations": state.iteration_count,
                "time_s": round(elapsed, 1),
                "ground_truth": item["answer"],
                "predicted_raw": predicted,
                "predicted_for_check": predicted_for_check,
                "final_verifier_decision": final_verifier_decision,
                "verifier_false_positive": final_verifier_decision == "CORRECT" and not correct,
                "verifier_false_negative": final_verifier_decision in ("MINOR_FLAW", "CRITICAL_FLAW") and correct,
                "run_status": state.status.value if state.status else None,
                "history": history_info,
            }
            status = "✅" if correct else "❌"
            print(f"  [{pid}] {status}  iters={state.iteration_count}  "
                  f"time={elapsed:.1f}s  gt={item['answer'][:30]}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
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
    verifier_fp = sum(1 for r in results if r.get("verifier_false_positive"))
    verifier_fn = sum(1 for r in results if r.get("verifier_false_negative"))
    partial_count = sum(1 for r in results if r.get("run_status") == "PARTIAL_PROGRESS")
    summary = {
        "dataset": "answerbench",
        "total": total,
        "correct": correct_count,
        "exact_match_accuracy": round(correct_count / total, 4) if total else 0.0,
        "partial_progress": partial_count,
        "verifier_false_positive": verifier_fp,
        "verifier_false_negative": verifier_fn,
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
        run_id = _build_run_id(pid)
        t0 = time.time()
        try:
            state = agent.solve(run_id, item["problem"], ground_truth=item.get("solution", ""))
            predicted = state.final_answer or state.current_proof
            completeness = check_proof_completeness(predicted)
            # 最终 Verifier 裁决（取最后一个 VERIFIER 历史项）
            verifier_entries = [e for e in state.history if e.agent_node == "VERIFIER"]
            final_decision = (
                verifier_entries[-1].decision.value if verifier_entries else "NO_VERDICT"
            )
            elapsed = time.time() - t0
            history_info = _serialize_history(state.history)
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
                "category": item.get("category", ""),
                "level": item.get("level", ""),
                "source": item.get("source", ""),
                "completeness": completeness,
                "final_verifier_decision": final_decision,
                "has_final_answer": state.final_answer is not None,
                "iterations": state.iteration_count,
                "time_s": round(elapsed, 1),
                "run_status": state.status.value if state.status else None,
                "history": history_info,
            }
            status = "✅" if state.final_answer is not None else "⚠️"
            print(f"  [{pid}] {status}  decision={final_decision}  "
                  f"iters={state.iteration_count}  time={elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
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
    partial_count = sum(1 for r in results if r.get("run_status") == "PARTIAL_PROGRESS")
    summary = {
        "dataset": "proofbench",
        "total": total,
        "format_complete": format_ok,
        "format_complete_rate": round(format_ok / total, 4) if total else 0.0,
        "correct_verdict": correct_v,
        "correct_verdict_rate": round(correct_v / total, 4) if total else 0.0,
        "has_final_answer": has_ans,
        "has_final_answer_rate": round(has_ans / total, 4) if total else 0.0,
        "partial_progress": partial_count,
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
    parser.add_argument(
        "--worklog-summary-mode",
        choices=["llm", "rule"],
        default="llm",
        help="工作日志阶段摘要模式：llm=调用模型二次摘要（默认），rule=规则摘要",
    )
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────
    from src.core.agent import AletheiaAgent
    from src.core.config import load_config, load_prompts
    from src.models.llm_client import create_llm_client
    from src.utils.data_loader import (
        load_answerbench_full,
        load_gradingbench_full,
        load_proofbench_full,
    )

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

    # ── 批量结束后按题生成新版 Markdown 工作日志 ────────────────────────
    if not args.no_worklog:
        from src.utils.worklog_builder import WorklogBuilder

        wb = WorklogBuilder(llm_config=config)
        generated_count = 0
        for dataset_results in all_results.values():
            for row in dataset_results:
                run_id = row.get("run_id")
                pid = row.get("problem_id")
                log_key = run_id or pid
                if not log_key:
                    continue
                jsonl_path = Path("data/logs") / f"{log_key}.jsonl"
                if not jsonl_path.exists():
                    continue
                md_path = Path("data/logs") / f"{log_key}.md"
                try:
                    wb.build_problem_worklog(str(jsonl_path), str(md_path))
                    generated_count += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠️  worklog 生成失败 {log_key}: {exc}")
        if generated_count:
            print(f"  📄 已生成 {generated_count} 份题目级工作日志（WorklogBuilder）。")


if __name__ == "__main__":
    main()
