"""General365 批量评测脚本：读 CSV 逐题运行通用推理分支并输出状态统计。"""

import argparse
from src.memory.state import RunStatus


def apply_limit(rows: list[dict], limit: int | None) -> list[dict]:
    """Return the first `limit` rows, or all rows if limit is None.

    Raises:
        ValueError: If limit is negative.
    """
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    return list(rows[:limit])


def state_to_status(state) -> str:
    """Map ProofState.status to SUCCESS/PROGRESS/FAILED string.

    Raises:
        ValueError: If status is None or unknown.
    """
    if state.status is None:
        raise ValueError("state.status is None")
    if state.status == RunStatus.SUCCESS:
        return "SUCCESS"
    if state.status == RunStatus.PROGRESS:
        return "PROGRESS"
    if state.status == RunStatus.FAILED:
        return "FAILED"
    raise ValueError(f"Unknown state.status: {state.status}")


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
        default=3,
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
