"""General365 批量评测脚本：读 CSV 逐题运行通用推理分支并输出状态统计。"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

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


def summarize_results(results: list[dict]) -> dict:
    """Aggregate problem results into summary statistics.

    Args:
        results: List of result dicts from build_problem_result().

    Returns:
        Dict with total/success/progress/failed counts and by_subcategory breakdown.

    Raises:
        ValueError: If a result has an unknown status or missing subcategory.
    """
    total = len(results)
    success = 0
    progress = 0
    failed = 0
    by_subcategory: dict[str, dict[str, int]] = {}

    for r in results:
        status = r.get("status")
        if status not in ("SUCCESS", "PROGRESS", "FAILED"):
            raise ValueError(f"Unknown status in result: {status}")

        if status == "SUCCESS":
            success += 1
        elif status == "PROGRESS":
            progress += 1
        elif status == "FAILED":
            failed += 1

        subcat = r.get("subcategory")
        if subcat is None:
            raise KeyError(f"Missing subcategory in result: {r.get('problem_id')}")
        bucket = by_subcategory.setdefault(subcat, {"total": 0, "SUCCESS": 0, "PROGRESS": 0, "FAILED": 0})
        bucket["total"] += 1
        bucket[status] += 1

    return {
        "total": total,
        "success": success,
        "progress": progress,
        "failed": failed,
        "by_subcategory": by_subcategory,
    }


def write_results(
    result: dict,
    output_dir: Path,
    dataset_path: Path,
    timestamp: str,
) -> tuple[Path, Path]:
    """Write result JSON and CSV to output_dir.

    Args:
        result: Full result dict (includes 'problems' list and summary stats).
        output_dir: Directory for output files.
        dataset_path: Original dataset path (used for filename stem).
        timestamp: Timestamp string for filenames.

    Returns:
        (json_path, csv_path)
    """
    dataset_stem = dataset_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"general365_{dataset_stem}_{timestamp}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / f"general365_{dataset_stem}_{timestamp}_details.csv"
    problems = result.get("problems", [])
    fieldnames = [
        "problem_id", "subcategory", "ground_truth", "status",
        "iteration_count", "error_type", "error_message", "run_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in problems:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return json_path, csv_path


class General365EvaluationRunner:
    """Runner that processes General365 problems through AletheiaAgent."""

    def __init__(
        self,
        agent,
        dataset_path: str = "data/problem-case/general365_selected_20.csv",
        max_turns: int | None = None,
        extract_stages: bool = True,
        runs_root: str = "runs",
        stage_extractor=None,
    ):
        self.agent = agent
        self.dataset_path = dataset_path
        self.max_turns = max_turns
        self.extract_stages = extract_stages
        self.runs_root = Path(runs_root)
        self.stage_extractor = stage_extractor

    def _run_single_problem(self, problem_entry: dict) -> dict:
        """Run a single problem through the agent and return a result dict.

        Raises:
            ValueError: If problem text is empty.
        """
        problem_text = (problem_entry.get("problem") or "").strip()
        if not problem_text:
            raise ValueError(f"Empty problem text for {problem_entry.get('problem_id', 'unknown')}")

        problem_id = problem_entry["problem_id"]
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{problem_id}_{run_ts}"

        result_dir = str(self.runs_root / run_id)

        try:
            state = self.agent.solve(
                problem_id=run_id,
                problem_text=problem_text,
                ground_truth=problem_entry.get("answer"),
            )
            final_status = state_to_status(state)
            result = build_problem_result(
                problem_entry, state, final_status, result_dir,
            )
        except Exception as exc:
            result = build_problem_result(
                problem_entry, None, "FAILED", result_dir,
            )
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)

        return result


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
