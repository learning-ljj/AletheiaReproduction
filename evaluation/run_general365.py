"""General365 批量评测脚本：读 CSV 逐题运行通用推理分支并输出状态统计。"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="General365 Batch Evaluation Runner",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/problem-case/general365_selected_20.csv",
        help="Path to General365 CSV dataset (default: data/problem-case/general365_selected_20.csv).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max refinement turns for each problem.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N problems (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for result JSON/CSV.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=False,
        help="Skip stage extraction for non-SUCCESS problems.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point (stub: only parser; Agent integration in later tasks)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f">>> Dataset: {args.dataset}")
    print(f">>> Max turns: {args.max_turns or 'default'}")
    print(f">>> Limit: {args.limit or 'all'}")
    print(">>> Runner not yet implemented (parser stub)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
