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


def build_problem_result(
    problem_entry: dict,
    state,
    final_status: str,
    run_dir: str | None,
    *,
    extraction_status: bool | None = None,
    extraction_error: str | None = None,
) -> dict:
    """Assemble a single problem result dict.

    Args:
        problem_entry: Row from load_general365_full().
        state: ProofState (or compatible object) from agent.solve().
        final_status: 'SUCCESS', 'PROGRESS', or 'FAILED'.
        run_dir: Run directory path string or None.
        extraction_status: Whether stage extraction succeeded (optional).
        extraction_error: Stage extraction error message (optional).

    Returns:
        Dict with result fields.
    """
    return {
        "problem_id": problem_entry["problem_id"],
        "category": problem_entry.get("category"),
        "subcategory": problem_entry.get("subcategory"),
        "source": problem_entry.get("source"),
        "ground_truth": problem_entry.get("answer"),
        "status": final_status,
        "iteration_count": getattr(state, "iteration_count", None),
        "run_dir": run_dir,
        "error_type": None,
        "error_message": None,
        "stage_extraction_status": extraction_status,
        "stage_extraction_error": extraction_error,
    }


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
