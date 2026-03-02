"""Aletheia CLI 入口：接受数学问题并运行 Generator→Verifier→Reviser 迭代精炼循环。"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.core.agent import AletheiaAgent
from src.core.config import load_config, load_prompts


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
        "--log", "-l",
        type=str,
        default=None,
        help="Path to save a human-readable log file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、加载配置、运行 Agent 并输出结果。返回 exit code。"""
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
    agent = AletheiaAgent(config, prompts)
    print(f">>> Problem ID: {problem_id}")
    print(f">>> Max turns: {agent.max_turns}")
    print(f">>> Running Aletheia Agent...\n")

    state = agent.solve(problem_id, problem_text)

    # ── 输出结果 ──
    print("\n" + "=" * 70)
    print(f">>> Iterations: {state.iteration_count}")
    print(f">>> Final answer: {'Found' if state.final_answer else 'Not found (max turns exhausted)'}")
    print("=" * 70)

    if state.final_answer:
        print("\n" + state.final_answer)

    # ── 可选：保存人类可读日志 ──
    if args.log:
        from src.utils.logger import write_readable_log
        log_path = Path(args.log)
        write_readable_log(
            problem_id=problem_id,
            problem_text=problem_text,
            history=state.history,
            final_answer=state.final_answer,
            filepath=log_path,
        )
        print(f"\n>>> Log saved to: {log_path}")

    print(f">>> JSONL logs saved to: data/logs/{problem_id}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
