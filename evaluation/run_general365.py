"""General365 批量评测脚本：读 CSV 逐题运行通用推理分支并输出状态统计。"""

import argparse
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
) -> Path:
    """Write result JSON to output_dir.

    Args:
        result: Full result dict (includes 'problems' list and summary stats).
        output_dir: Directory for output files.
        dataset_path: Original dataset path (used for filename stem).
        timestamp: Timestamp string for filenames.

    Returns:
        json_path
    """
    dataset_stem = dataset_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"general365_{dataset_stem}_{timestamp}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def print_summary(stats: dict) -> None:
    """Print summary statistics to terminal."""
    print(f"\n{'=' * 50}")
    print(f"Total:   {stats['total']}")
    print(f"SUCCESS: {stats['success']}")
    print(f"PROGRESS: {stats['progress']}")
    print(f"FAILED:  {stats['failed']}")
    if stats.get("by_subcategory"):
        print()
        for subcat, counts in stats["by_subcategory"].items():
            print(f"  {subcat}: {counts['total']} total "
                  f"(SUCCESS={counts['SUCCESS']}, PROGRESS={counts['PROGRESS']}, FAILED={counts['FAILED']})")
    print(f"{'=' * 50}\n")


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

        If extract_stages is True and status is not SUCCESS, attempts to extract
        intermediate stages from history.jsonl.

        Raises:
            ValueError: If problem text is empty.
        """
        problem_text = (problem_entry.get("problem") or "").strip()
        if not problem_text:
            raise ValueError(f"Empty problem text for {problem_entry.get('problem_id', 'unknown')}")

        problem_id = problem_entry["problem_id"]
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{problem_id}_{run_ts}"

        result_dir_path = self.runs_root / run_id
        result_dir = str(result_dir_path)

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
            final_status = "FAILED"

        # Extract stages for non-SUCCESS problems
        if self.extract_stages and final_status != "SUCCESS":
            history_path = result_dir_path / "history.jsonl"
            if history_path.exists():
                extracted_dir = result_dir_path / "extracted"
                try:
                    if self.stage_extractor is None:
                        from evaluation.extract_stages import extract_stages as default_extractor
                        self.stage_extractor = default_extractor
                    self.stage_extractor(str(history_path), str(extracted_dir))
                except Exception as exc:
                    result["stage_extraction_error"] = str(exc)

        return result

    def run_all(self, limit: int | None = None) -> dict:
        """Run all problems from the dataset and return aggregated results.

        Args:
            limit: Optional cap on number of problems to run.

        Returns:
            Dict with dataset metadata, summary stats, and per-problem results.
        """
        from evaluation.data_loader import load_general365_full

        all_rows = load_general365_full(self.dataset_path)
        rows = apply_limit(all_rows, limit)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        problems: list[dict] = []

        for i, entry in enumerate(rows):
            pid = entry.get("problem_id", f"row_{i}")
            subcat = entry.get("subcategory", "unknown")
            print(f"[{i + 1}/{len(rows)}] {pid} ({subcat})...", end=" ", flush=True)

            result = self._run_single_problem(entry)
            problems.append(result)
            print(f"{result['status']}")

        stats = summarize_results(problems)
        result = {
            "dataset": "general365",
            "dataset_path": str(self.dataset_path),
            "timestamp": timestamp,
            **stats,
            "problems": problems,
        }

        print_summary(stats)
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
    """Main entry point: parse args, instantiate Agent, run all, write results."""
    from src.core.agent import AletheiaAgent
    from src.core.config import load_config, load_prompts

    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config()
    prompts = load_prompts("config/prompts_general")

    agent = AletheiaAgent(config, prompts)
    runner = General365EvaluationRunner(
        agent=agent,
        dataset_path=args.dataset,
        max_turns=args.max_turns,
        runs_root=config.get("agent", {}).get("runs_root", "runs"),
    )

    result = runner.run_all(limit=args.limit)

    output_dir = Path(args.output_dir) if args.output_dir else Path("results/general365")
    timestamp = result["timestamp"]
    json_path = write_results(result, output_dir, Path(args.dataset), timestamp)
    print(f"Results saved to: {json_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
