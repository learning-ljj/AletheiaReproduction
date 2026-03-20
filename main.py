"""Aletheia CLI 入口：接受数学问题并运行 Generator→Verifier→Reviser 迭代精炼循环。"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.core.agent import AletheiaAgent
from src.core.config import load_config, load_prompts
from src.core.state import RunStatus
from src.utils.data_loader import lookup_ground_truth
from src.utils.worklog_builder import WorklogBuilder


def _configure_stdio_utf8() -> None:
    """统一 stdout/stderr 为 UTF-8，避免 Windows 重定向日志乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                continue


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Aletheia — Iterative Refinement Agent for Mathematical Reasoning",
    )
    parser.add_argument(
        "problem_file",
        nargs="?",
        default=None,
        help="Path to a text file containing the problem statement.",
    )
    parser.add_argument(
        "--problem", "-p",
        type=str,
        default=None,
        help="Problem statement as inline text (alternative to file).",
    )
    parser.add_argument(
        "--max-turns", "-m",
        type=int,
        default=None,
        help="Override max refinement turns (default: from settings.yaml).",
    )
    parser.add_argument(
        "--generate-worklog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate markdown worklog after run (default: true).",
    )
    parser.add_argument(
        "--worklog-path",
        type=str,
        default=None,
        help="Optional output path for markdown worklog (default: data/logs/{problem_id}.md).",
    )
    return parser


def _maybe_build_worklog(problem_id: str, worklog_path: str | None = None, llm_config: dict | None = None) -> str | None:
    """若 raw 日志存在则生成 markdown 工作日志，返回输出路径。"""
    run_jsonl_path = Path("data/logs") / f"{problem_id}.jsonl"
    if not run_jsonl_path.exists():
        return None

    output_md = Path(worklog_path) if worklog_path else (Path("data/logs") / f"{problem_id}.md")
    wb = WorklogBuilder(llm_config=llm_config)
    wb.build_problem_worklog(str(run_jsonl_path), str(output_md))
    return str(output_md)


def main(argv: list[str] | None = None) -> int:
    """解析参数、加载配置、运行 Agent 并输出结果。返回 exit code。"""
    _configure_stdio_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── 确定问题文本 ──
    if args.problem:
        problem_text = args.problem
        problem_id = "inline"
    elif args.problem_file:
        path = Path(args.problem_file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        problem_text = path.read_text(encoding="utf-8").strip()
        problem_id = path.stem
    else:
        parser.print_help()
        return 1

    # ── 加载配置 ──
    config = load_config()
    prompts = load_prompts()

    # ── 覆盖 max_turns ──
    if args.max_turns is not None:
        config.setdefault("agent", {})["max_turns"] = args.max_turns

    # ── 初始化并运行 Agent ──
    # 生成带时间戳的唯一运行 ID，避免多次运行的 JSONL 日志相互覆盖。
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{problem_id}_{run_ts}"

    agent = AletheiaAgent(config, prompts)
    ground_truth, gt_source = lookup_ground_truth(problem_id)
    print(f">>> Problem ID: {problem_id}")
    print(f">>> Run ID:     {run_id}")
    print(f">>> Max turns: {agent.max_turns}")
    if ground_truth:
        print(f">>> Ground truth loaded from: {gt_source}")
    print(f">>> Running Aletheia Agent...\n")

    state = None
    solve_error: Exception | None = None
    try:
        state = agent.solve(run_id, problem_text, ground_truth=ground_truth)
    except Exception as exc:  # noqa: BLE001
        solve_error = exc
        print(f"\n>>> Solve error: {exc}", file=sys.stderr)

    # ── 输出结果 ──
    if state is not None:
        print("\n" + "=" * 70)
        print(f">>> Iterations: {state.iteration_count}")
        if state.status == RunStatus.SUCCESS:
            print(">>> Result: SUCCESS — Complete solution found")
        elif state.status == RunStatus.PARTIAL:
            print(">>> Result: PARTIAL — Meaningful progress made (partial solution)")
        elif state.status == RunStatus.FAILED:
            reason = state.failure_reason or "unknown"
            print(f">>> Result: FAILED ({reason})")
        print("=" * 70)

        if state.final_output:
            print("\n" + state.final_output)

    # ── 可选：生成 markdown 工作日志（即使 solve 异常也尝试） ──
    if args.generate_worklog:
        try:
            generated = _maybe_build_worklog(
                problem_id=run_id,
                worklog_path=args.worklog_path,
                llm_config=config,
            )
            if generated:
                print(f">>> Worklog markdown saved to: {generated}")
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error("Worklog generation failed", exc_info=True)
            print(f">>> Worklog generation failed: {exc}", file=sys.stderr)

    print(f">>> JSONL logs saved to: data/logs/{run_id}.jsonl")
    return 0 if solve_error is None else 1


if __name__ == "__main__":
    sys.exit(main())
