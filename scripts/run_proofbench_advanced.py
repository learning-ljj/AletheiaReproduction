"""Run only PB-Advanced problems from proofbench.

鐢ㄦ硶绀轰緥锛?
  python scripts/run_proofbench_advanced.py --count 1 --max-turns 3
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root is on sys.path so `from src...` imports work when
# running the script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import AletheiaAgent
from dotenv import load_dotenv

# Load environment variables from .env so config can pick up API keys.
load_dotenv()
from src.utils.logging.worklog_builder import WorklogBuilder
from src.core.config import load_config, load_prompts
from src.utils.evaluation.data_loader import load_proofbench_full
from src.utils.evaluation.evaluator import check_proof_completeness
from src.utils.logging.raw_log_reader import load_raw_events, resolve_run_artifact_path, resolve_run_log_path
from src.utils.parsing.parser import parse_lemmas


def _normalize_lemma_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _last_verifier_decision(events: list[dict]) -> str | None:
    verifier_events = [
        item for item in events
        if str(item.get("node", "")).upper() == "VERIFIER"
    ]
    if not verifier_events:
        return None
    decision = verifier_events[-1].get("decision")
    return str(decision) if decision is not None else None


def _collect_run_quality_metrics(run_id: str, runs_root: Path) -> dict:
    metric = {
        "lemma_candidate_count": 0,
        "verified_lemma_count": 0,
        "lemma_accept_rate": 0.0,
        "citation_warning_count": 0,
        "has_citation_warning": False,
        "run_log_path": str(resolve_run_log_path(problem_id=run_id, runs_root=runs_root)),
    }

    try:
        events = load_raw_events(problem_id=run_id, runs_root=runs_root)
    except Exception as exc:  # noqa: BLE001
        metric["run_log_error"] = str(exc)
        return metric

    candidate_lemmas: set[str] = set()
    verified_lemmas: set[str] = set()
    citation_warning_count = 0

    for event in events:
        node = str(event.get("node", "")).upper()

        if node in {"GENERATOR", "REVISER"}:
            content = str(event.get("content", "") or "")
            for lemma in parse_lemmas(content):
                normalized = _normalize_lemma_text(lemma)
                if normalized:
                    candidate_lemmas.add(normalized)
            continue

        if node == "VERIFIER":
            verified_items = event.get("verified_lemmas") or []
            if isinstance(verified_items, str):
                verified_items = [verified_items]
            if isinstance(verified_items, list):
                for lemma in verified_items:
                    normalized = _normalize_lemma_text(lemma)
                    if normalized and normalized.upper() != "NONE":
                        verified_lemmas.add(normalized)
            continue

        if node == "WARNING":
            warning_type = str(event.get("warning_type", "")).lower()
            warning_text = str(event.get("warning", "")).lower()
            if warning_type == "citation_review" or "citation" in warning_text:
                citation_warning_count += 1

    candidate_count = len(candidate_lemmas)
    verified_count = len(verified_lemmas)
    metric.update({
        "lemma_candidate_count": candidate_count,
        "verified_lemma_count": verified_count,
        "lemma_accept_rate": round(verified_count / candidate_count, 4) if candidate_count else 0.0,
        "citation_warning_count": citation_warning_count,
        "has_citation_warning": citation_warning_count > 0,
    })
    return metric


def run_proofbench(agent, data: list[dict], runs_root: Path):
    results = []
    for item in data:
        pid = item["problem_id"]
        run_id = f"{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        t0 = time.time()
        try:
            state = agent.solve(run_id, item["problem"], ground_truth=item.get("solution", ""))
            predicted = state.final_answer or state.current_proof
            completeness = check_proof_completeness(predicted)
            try:
                events = load_raw_events(problem_id=run_id, runs_root=runs_root)
            except Exception:
                events = []
            final_decision = _last_verifier_decision(events) or "NO_VERDICT"
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
                "completeness": completeness,
                "final_verifier_decision": final_decision,
                "has_final_answer": state.final_answer is not None,
                "iterations": getattr(state, "iteration_count", 0),
                "time_s": round(elapsed, 1),
            }
            entry.update(_collect_run_quality_metrics(run_id=run_id, runs_root=runs_root))
            print(f"  [{pid}] iters={entry['iterations']} decision={final_decision} time={elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "problem_id": pid,
                "run_id": run_id,
                "problem": item.get("problem", ""),
                "completeness": {},
                "final_verifier_decision": "ERROR",
                "has_final_answer": False,
                "iterations": 0,
                "time_s": round(elapsed, 1),
                "error": str(exc),
            }
            entry.update(_collect_run_quality_metrics(run_id=run_id, runs_root=runs_root))
            print(f"  [{pid}] 鈿狅笍  Error: {exc}")
        results.append(entry)

    total = len(results)
    format_ok = sum(1 for r in results if r.get("completeness", {}).get("has_preliminary_solution"))
    correct_v = sum(1 for r in results if r.get("final_verifier_decision") == "CORRECT")
    has_ans = sum(1 for r in results if r.get("has_final_answer"))
    total_lemma_candidates = sum(int(r.get("lemma_candidate_count", 0) or 0) for r in results)
    total_verified_lemmas = sum(int(r.get("verified_lemma_count", 0) or 0) for r in results)
    citation_warning_count = sum(int(r.get("citation_warning_count", 0) or 0) for r in results)
    citation_warning_problems = sum(1 for r in results if r.get("has_citation_warning"))
    summary = {
        "dataset": "proofbench_advanced",
        "total": total,
        "format_complete": format_ok,
        "format_complete_rate": round(format_ok / total, 4) if total else 0.0,
        "correct_verdict": correct_v,
        "correct_verdict_rate": round(correct_v / total, 4) if total else 0.0,
        "has_final_answer": has_ans,
        "has_final_answer_rate": round(has_ans / total, 4) if total else 0.0,
        "total_lemma_candidates": total_lemma_candidates,
        "total_verified_lemmas": total_verified_lemmas,
        "lemma_accept_rate": round(total_verified_lemmas / total_lemma_candidates, 4)
        if total_lemma_candidates else 0.0,
        "citation_warning_count": citation_warning_count,
        "citation_warning_problems": citation_warning_problems,
        "citation_warning_rate": round(citation_warning_problems / total, 4) if total else 0.0,
        "error_count": sum(1 for r in results if "error" in r),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PB-Advanced problems only")
    parser.add_argument("--count", type=int, default=1, help="Number of PB-Advanced problems to run (0 = all)")
    parser.add_argument("--max-turns", type=int, default=3, help="Agent max turns per problem")
    parser.add_argument("--output-dir", type=str, default="runs/benchmarks", help="Output directory for JSON results")
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()
    config.setdefault("agent", {})["max_turns"] = args.max_turns
    runs_root = Path(config.get("agent", {}).get("runs_root", "runs"))

    full = load_proofbench_full()
    adv = [d for d in full if (d.get("problem_id") or "").startswith("PB-Advanced")]
    if args.count > 0:
        adv = adv[: args.count]

    if not adv:
        print("No PB-Advanced items found in data/imobench/proofbench.csv")
        return

    agent = AletheiaAgent(config, prompts)
    results, summary = run_proofbench(agent, adv, runs_root=runs_root)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"imobench_proofbench_advanced_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")
    # 灏濊瘯涓烘瘡涓?run_id 鐢熸垚 Markdown 宸ヤ綔鏃ュ織锛堢绾跨敓鎴愶級
    try:
        wb = WorklogBuilder(llm_config=config)
        generated = 0
        for row in results:
            run_id = row.get("run_id")
            if not run_id:
                continue
            jsonl_path = resolve_run_log_path(problem_id=run_id, runs_root=runs_root)
            if not jsonl_path.exists():
                continue
            md_path = resolve_run_artifact_path(
                problem_id=run_id,
                artifact_name="worklog.md",
                runs_root=runs_root,
            )
            md_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                wb.build_problem_worklog(str(jsonl_path), str(md_path))
                generated += 1
            except Exception:
                # 鍗曚釜 worklog 澶辫触涓嶅奖鍝嶆€讳綋缁撴灉
                continue
        if generated:
            print(f"Generated {generated} worklog(s) to {runs_root}/<run_id>/artifact/")
    except Exception as exc:  # noqa: BLE001
        print(f"Worklog generation skipped or failed: {exc}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

