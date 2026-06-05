r"""
General365 通用推理批量评测脚本 —— 运行与统计整合版本。

本脚本将评测执行和结果统计整合在单一文件中，支持 General365 通用推理数据集。
CSV 数据集的路径可通过命令行参数配置。

定义的三种最终状态：
  - SUCCESS:   Verifier 认为模型成功解决问题（state.status == RunStatus.SUCCESS）
  - PROGRESS:  轮次耗尽但存在部分进展（iteration_count >= max_turns 但未完成）
  - FAILED:    轮次耗尽且无有效进展（iteration_count >= max_turns 且停滞）

对于非 SUCCESS 的题目，会自动调用 extract_stages.py 提取模型中各阶段的输出，
并将 LaTeX 显示数学标记 \[ ... \] 规范化为 Markdown 通用的 $$ ... $$ 格式。

用法示例：
    # 运行全部题目
    python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --max-turns 3

    # 仅运行前 5 题
    python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --limit 5

    # 不提取非 SUCCESS 题目的模型输出
    python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --no-extract

    # 指定自定义输出目录
    python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --output-dir my_results

输出：
    results/general365/general365_{dataset_filename}_{timestamp}.json

数据集格式：
    CSV 文件需包含以下列：
    - problem_id:   问题唯一标识（必填）
    - problem:      问题文本（必填）
    - answer:       标准答案 / ground truth（必填）
    - category:     大类（可选）
    - subcategory:  子类（可选）
    - source:       来源（可选）

提取内容示例：
    - 提取位置: runs/{problem_id}_{timestamp}/extracted/
    - 包含文件: 0_generator.md, 0_generator_reasoning.md, ...
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.memory.state import RunStatus


def apply_limit(rows: list[dict], limit: int | None) -> list[dict]:
    """返回数据集的前 `limit` 行，若 limit 为 None 则返回全部。

    Raises:
        ValueError: 如果 limit 为负数。

    Args:
        rows:  数据集行列表，每行为一个 problem 条目 dict。
        limit: 需要截取的行数，None 表示不限。

    Returns:
        截取后的行列表。
    """
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError(f"limit 不能为负数，实际值: {limit}")
    return list(rows[:limit])


def state_to_status(state) -> str:
    """将 ProofState.status 映射为 SUCCESS / PROGRESS / FAILED 字符串。

    Raises:
        ValueError: 如果 status 为 None 或无法识别的值。

    Args:
        state: ProofState 对象（或任何包含 .status 属性的兼容对象）。

    Returns:
        "SUCCESS"、"PROGRESS" 或 "FAILED"。
    """
    if state.status is None:
        raise ValueError("state.status 为 None，无法映射状态")
    if state.status == RunStatus.SUCCESS:
        return "SUCCESS"
    if state.status == RunStatus.PROGRESS:
        return "PROGRESS"
    if state.status == RunStatus.FAILED:
        return "FAILED"
    raise ValueError(f"未知的 state.status: {state.status}")


def build_problem_result(
    problem_entry: dict,
    state,
    final_status: str,
    run_dir: str | None,
    *,
    extraction_status: bool | None = None,
    extraction_error: str | None = None,
) -> dict:
    """组装单个题目的结果字典。

    将 agent 运行后的状态信息、数据集元数据以及提取状态整合为一个结构化结果。

    Args:
        problem_entry:  从 load_general365_full() 读取的问题数据行。
        state:          agent.solve() 返回的 ProofState（或兼容对象）。
        final_status:   'SUCCESS'、'PROGRESS' 或 'FAILED'。
        run_dir:        运行目录路径字符串，None 表示无目录。
        extraction_status: 阶段提取是否成功（可选）。
        extraction_error:  阶段提取错误信息（可选）。

    Returns:
        包含问题元数据、运行状态和提取状态的结果字典。
    """
    return {
        "problem_id": problem_entry["problem_id"],       # 问题唯一标识
        "category": problem_entry.get("category"),        # 大类（如 algebra）
        "subcategory": problem_entry.get("subcategory"),  # 子类（如 polynomial）
        "source": problem_entry.get("source"),            # 来源（如 IMO Shortlist）
        "ground_truth": problem_entry.get("answer"),      # 标准答案
        "status": final_status,                           # 最终状态
        "iteration_count": getattr(state, "iteration_count", None),  # 运行轮次
        "run_dir": run_dir,                               # 运行产物目录
        "error_type": None,                               # 异常类型（如有）
        "error_message": None,                            # 异常信息（如有）
        "stage_extraction_status": extraction_status,     # 阶段提取状态
        "stage_extraction_error": extraction_error,       # 阶段提取错误信息
    }


def summarize_results(results: list[dict]) -> dict:
    """汇总所有题目的运行结果，生成统计摘要。

    统计 SUCCESS / PROGRESS / FAILED 的数量，并按 subcategory 分组展示。

    Args:
        results: build_problem_result() 生成的结果字典列表。

    Returns:
        包含总数和按子类分组的统计字典。

    Raises:
        ValueError: 如果结果中存在未知的 status 值。
        KeyError:   如果结果中缺少 subcategory 字段。
    """
    total = len(results)
    success = 0
    progress = 0
    failed = 0
    # 按子类分组的统计桶，格式: {subcat: {"total": N, "SUCCESS": N, ...}}
    by_subcategory: dict[str, dict[str, int]] = {}

    for r in results:
        status = r.get("status")
        if status not in ("SUCCESS", "PROGRESS", "FAILED"):
            raise ValueError(f"结果中存在未知状态: {status}")

        if status == "SUCCESS":
            success += 1
        elif status == "PROGRESS":
            progress += 1
        elif status == "FAILED":
            failed += 1

        subcat = r.get("subcategory")
        if subcat is None:
            raise KeyError(f"结果缺少 subcategory 字段: {r.get('problem_id')}")
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
    """将完整的运行结果写入 JSON 文件。

    输出文件路径格式: {output_dir}/general365_{dataset_stem}_{timestamp}.json

    Args:
        result:      完整的结果字典（包含 'problems' 列表和统计摘要）。
        output_dir:  输出目录路径。
        dataset_path: 原始数据集路径（用于生成文件名主干）。
        timestamp:   时间戳字符串（用于文件名）。

    Returns:
        写入的 JSON 文件路径。
    """
    dataset_stem = dataset_path.stem  # 数据集文件名（不含扩展名）
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"general365_{dataset_stem}_{timestamp}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def print_summary(stats: dict) -> None:
    """在终端打印统计摘要。

    Args:
        stats: summarize_results() 返回的统计字典。
    """
    print(f"\n{'=' * 50}")
    print(f"总计:    {stats['total']}")
    print(f"SUCCESS: {stats['success']}")
    print(f"PROGRESS: {stats['progress']}")
    print(f"FAILED:  {stats['failed']}")
    if stats.get("by_subcategory"):
        print()
        for subcat, counts in stats["by_subcategory"].items():
            print(f"  {subcat}: {counts['total']} 题 "
                  f"(SUCCESS={counts['SUCCESS']}, PROGRESS={counts['PROGRESS']}, FAILED={counts['FAILED']})")
    print(f"{'=' * 50}\n")


class General365EvaluationRunner:
    """General365 批量评测运行器。

    通过 AletheiaAgent 逐个处理数据集中的题目，收集运行结果，
    并为非 SUCCESS 的题目自动提取中间阶段的模型输出（含 LaTeX 数学公式规范化）。

    Attributes:
        agent:          AletheiaAgent 实例，提供 solve() 接口。
        dataset_path:   CSV 数据集文件路径。
        max_turns:      每个问题的最大精炼轮次。
        extract_stages: 是否为非 SUCCESS 题目提取阶段输出。
        runs_root:      运行产物根目录。
        stage_extractor: 阶段提取函数，默认使用 evaluation.extract_stages.extract_stages。
    """

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
        r"""运行单个题目，返回结果字典。

        调用 agent.solve() 执行推理，若 extract_stages=True 且最终状态不是 SUCCESS，
        则自动调用 extract_stages 从 history.jsonl 中提取各阶段输出，
        并应用 LaTeX 数学公式规范化（\[ ... \] → $$ ... $$）。

        Raises:
            ValueError: 如果问题文本为空。

        Args:
            problem_entry: 包含 problem_id、problem、answer 等字段的数据行。

        Returns:
            build_problem_result() 生成的结果字典。
        """
        # 验证问题文本不为空
        problem_text = (problem_entry.get("problem") or "").strip()
        if not problem_text:
            raise ValueError(f"问题文本为空: {problem_entry.get('problem_id', 'unknown')}")

        problem_id = problem_entry["problem_id"]
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{problem_id}_{run_ts}"  # 运行标识 = 问题ID_时间戳

        result_dir_path = self.runs_root / run_id
        result_dir = str(result_dir_path)

        try:
            # 调用 agent 求解
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
            # agent 执行异常时，标记为 FAILED 并记录错误信息
            result = build_problem_result(
                problem_entry, None, "FAILED", result_dir,
            )
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
            final_status = "FAILED"

        # 对非 SUCCESS 题目，提取中间阶段输出以供分析
        if self.extract_stages and final_status != "SUCCESS":
            history_path = result_dir_path / "history.jsonl"
            if history_path.exists():
                extracted_dir = result_dir_path / "extracted"
                try:
                    # 延迟导入默认提取器，避免循环依赖
                    if self.stage_extractor is None:
                        from evaluation.extract_stages import extract_stages as default_extractor
                        self.stage_extractor = default_extractor
                    # extract_stages 内部已调用 normalize_latex_display_math
                    # 自动将 \[ ... \] 规范化为 $$ ... $$
                    self.stage_extractor(str(history_path), str(extracted_dir))
                except Exception as exc:
                    result["stage_extraction_error"] = str(exc)

        return result

    def run_all(self, limit: int | None = None) -> dict:
        """运行数据集中的所有题目，返回聚合结果。

        逐题调用 _run_single_problem()，收集结果后生成统计摘要。

        Args:
            limit: 可选，仅运行前 N 道题。

        Returns:
            包含数据集元数据、统计摘要和逐题结果的字典。
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

        # 生成统计摘要
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
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        description="General365 批量评测运行器 — 执行 General365 通用推理评估并统计结果",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/problem-case/general365_selected_20.csv",
        help="General365 CSV 数据集路径（默认: data/problem-case/general365_selected_20.csv）。",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="每个问题的最大精炼轮次（默认: 3）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 道题（默认: 全部运行）。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果 JSON 输出目录（默认: results/general365）。",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=False,
        help="跳过对非 SUCCESS 题目的阶段输出提取。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """主入口：解析参数 → 加载配置 → 实例化 Agent → 运行评测 → 写入结果。

    流程说明：
    1. 从 .env 加载环境变量（LLM_PROVIDER、API Key 等）
    2. 加载 settings.yaml 配置（其中 ${VAR} 占位符会被环境变量替换）
    3. 加载 prompts_general 目录下的提示词模板
    4. 创建 AletheiaAgent 实例
    5. 创建 General365EvaluationRunner 并运行评测
    6. 将结果写入 JSON 文件
    7. 提取非 SUCCESS 题目的阶段输出（含 LaTeX 公式规范化）

    Args:
        argv: 命令行参数列表（默认使用 sys.argv）。

    Returns:
        退出码（0 表示成功）。
    """
    from dotenv import load_dotenv

    # 加载 .env 文件，使 ${LLM_PROVIDER} 等占位符能被正确替换
    load_dotenv()

    from src.core.agent import AletheiaAgent
    from src.core.config import load_config, load_prompts

    parser = build_parser()
    args = parser.parse_args(argv)

    # 加载配置和提示词模板
    config = load_config()
    prompts = load_prompts("config/prompts_general")

    # 创建 Agent 实例
    agent = AletheiaAgent(config, prompts)
    runner = General365EvaluationRunner(
        agent=agent,
        dataset_path=args.dataset,
        max_turns=args.max_turns,
        # 从配置中读取 runs_root，默认为 "runs"
        runs_root=config.get("agent", {}).get("runs_root", "runs"),
    )

    # 执行评测
    result = runner.run_all(limit=args.limit)

    # 写入结果文件
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/general365")
    timestamp = result["timestamp"]
    json_path = write_results(result, output_dir, Path(args.dataset), timestamp)
    print(f"结果已保存至: {json_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
