"""IMOBench 批量评测脚本：运行四个 benchmark 数据集，统计成功/失败。

支持的数据集：
  - answerbench: 短答题
  - proofbench: 证明题
  - gradingbench: 评分题

用法：
    python -m evaluation.run_imobench --dataset answerbench [--count 10] [--max-turns 3]
    python -m evaluation.run_imobench --dataset proofbench [--count 10] [--max-turns 3]
    python -m evaluation.run_imobench --dataset gradingbench [--count 10] [--max-turns 3]
    python -m evaluation.run_imobench --dataset all [--count 10]

输出格式（控制台 + JSON）：
    逐题结果 + 汇总统计表格
    evaluation/results/imobench_{dataset}_{timestamp}.json

评测指标（不调用评分模型）：
  - TOTAL: 总题数
  - SUCCESS: state.status == RunStatus.SUCCESS 的题数
  - FAILURE: 其他状态的题数
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 加载环境变量并配置 UTF-8
load_dotenv()

from src.core.agent import AletheiaAgent
from src.core.config import load_config, load_prompts
from src.memory.state import RunStatus
from evaluation.data_loader import (
    load_answerbench_full,
    load_proofbench_full,
    load_gradingbench_full,
    lookup_ground_truth,
)


def _configure_stdio_utf8() -> None:
    """统一 stdout/stderr 为 UTF-8，避免 Windows 重定向日志乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                continue


class BenchmarkRunner:
    """单个 benchmark 的评测运行器。高内聚低耦合的设计：
    - 职责单一：加载数据集、运行 Agent、收集结果
    - 错误直接暴露，不做兜底性处理
    """

    def __init__(
        self,
        agent: AletheiaAgent,
        dataset_name: Literal["answerbench", "proofbench", "gradingbench"],
        max_count: int | None = None,
        max_turns: int | None = None,
    ):
        """初始化评测运行器。

        Args:
            agent: Aletheia Agent 实例
            dataset_name: 数据集名称
            max_count: 最多评测多少题（None 表示全部）
            max_turns: 单题最大轮次
        """
        self.agent = agent
        self.dataset_name = dataset_name
        self.max_count = max_count
        self.max_turns = max_turns or agent.max_turns
        
        # 覆盖 agent 的 max_turns 配置
        if max_turns is not None:
            self.agent.orchestrator.max_turns = max_turns

        # 数据集加载函数映射
        self._loaders = {
            "answerbench": load_answerbench_full,
            "proofbench": load_proofbench_full,
            "gradingbench": load_gradingbench_full,
        }

    def load_dataset(self) -> list[dict]:
        """加载指定 benchmark 数据集。错误直接暴露。"""
        loader = self._loaders[self.dataset_name]
        dataset = loader()
        
        if self.max_count is not None:
            dataset = dataset[: self.max_count]
        
        return dataset

    def _run_single_problem(self, problem_entry: dict) -> dict:
        """运行单个问题。返回 {problem_id, status, error}。"""
        problem_id = problem_entry.get("problem_id", "unknown")
        problem_text = problem_entry.get("problem", "")
        
        if not problem_text.strip():
            return {
                "problem_id": problem_id,
                "status": "SKIP",
                "error": "empty problem text",
            }
        
        try:
            # 回填 ground_truth（仅用于记录，不用于评分）
            ground_truth, _ = lookup_ground_truth(problem_id)
            
            # 运行 Agent
            state = self.agent.solve(problem_id, problem_text, ground_truth=ground_truth)
            
            # 记录最终状态
            status_str = state.status.value if state.status else "UNKNOWN"
            return {
                "problem_id": problem_id,
                "status": status_str,
                "iteration_count": state.iteration_count,
                "error": None,
            }
        except Exception as exc:
            # 错误直接暴露：调用方可以选择继续或中止
            return {
                "problem_id": problem_id,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:100]}",
            }

    def run_all(self) -> dict:
        """运行整个 benchmark，返回统计结果。

        Returns:
            {
                "dataset": str,
                "total": int,
                "success": int,
                "failure": int,
                "problems": list[dict],  # 每题的结果
            }
        """
        dataset = self.load_dataset()
        total = len(dataset)
        results = []
        success_count = 0
        failure_count = 0

        print(f"\n{'=' * 70}")
        print(f"Benchmark: {self.dataset_name.upper()}")
        print(f"Total problems: {total}, Max turns: {self.max_turns}")
        print(f"{'=' * 70}\n")

        for idx, problem_entry in enumerate(dataset, start=1):
            problem_id = problem_entry.get("problem_id", "unknown")
            print(f"[{idx:3d}/{total}] {problem_id:40s} ", end="", flush=True)
            
            result = self._run_single_problem(problem_entry)
            status = result.get("status")
            
            # 统计：只有 SUCCESS 算成功，其他都算失败或跳过
            if status == RunStatus.SUCCESS.value:
                success_count += 1
                print(f"✓ SUCCESS")
            elif status == "SKIP":
                print(f"⊘ SKIP ({result['error']})")
                failure_count += 1
            elif status == "ERROR":
                print(f"✗ ERROR: {result['error']}")
                failure_count += 1
            else:
                # PROGRESS, FAILED, 等其他状态都算失败
                print(f"✗ {status}")
                failure_count += 1
            
            results.append(result)

        print(f"\n{'─' * 70}")
        print(f"Summary: SUCCESS={success_count}, FAILURE={failure_count}, TOTAL={total}")
        print(f"Success rate: {100.0 * success_count / total:.1f}%")
        print(f"{'─' * 70}\n")

        return {
            "dataset": self.dataset_name,
            "total": total,
            "success": success_count,
            "failure": failure_count,
            "problems": results,
        }


def run_benchmark(
    dataset_name: Literal["answerbench", "proofbench", "gradingbench"],
    agent: AletheiaAgent,
    max_count: int | None = None,
    max_turns: int | None = None,
) -> dict:
    """运行单个 benchmark。

    Args:
        dataset_name: 数据集名称
        agent: Aletheia Agent 实例
        max_count: 最多评测多少题
        max_turns: 最大轮次

    Returns:
        {dataset, total, success, failure, problems}
    """
    runner = BenchmarkRunner(agent, dataset_name, max_count, max_turns)
    return runner.run_all()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="IMOBench 批量评测脚本",
    )
    parser.add_argument(
        "--dataset",
        choices=["answerbench", "proofbench", "gradingbench", "all"],
        default="all",
        help="要评测的数据集（默认：全部）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="最多评测多少题（默认：全部）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="单题最大轮次（默认：从配置读取）",
    )
    return parser


def save_results(results: dict | list[dict], output_dir: Path = None) -> Path:
    """保存评测结果到 JSON 文件。

    Args:
        results: 单个或多个评测结果
        output_dir: 输出目录（默认：evaluation/results）

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = Path("evaluation/results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定文件名
    if isinstance(results, list):
        filename = f"imobench_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        dataset = results.get("dataset", "unknown")
        filename = f"imobench_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    filepath = output_dir / filename
    filepath.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return filepath


def main(argv: list[str] | None = None) -> int:
    """主入口：解析参数、初始化 Agent、运行评测、保存结果。"""
    _configure_stdio_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    # 加载配置和提示词
    try:
        config = load_config()
        prompts = load_prompts()
    except Exception as exc:
        print(f"Failed to load config/prompts: {exc}", file=sys.stderr)
        return 1

    # 初始化 Agent
    try:
        agent = AletheiaAgent(config, prompts)
    except Exception as exc:
        print(f"Failed to initialize agent: {exc}", file=sys.stderr)
        return 1

    # 确定要运行的数据集
    datasets = (
        ["answerbench", "proofbench", "gradingbench"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_results = []
    
    try:
        for dataset_name in datasets:
            result = run_benchmark(
                dataset_name=dataset_name,  # type: ignore
                agent=agent,
                max_count=args.count,
                max_turns=args.max_turns,
            )
            all_results.append(result)
    except Exception as exc:
        print(f"\nBenchmark execution failed: {exc}", file=sys.stderr)
        return 1

    # 保存结果
    try:
        output_path = save_results(all_results if len(all_results) > 1 else all_results[0])
        print(f"Results saved to: {output_path}")
    except Exception as exc:
        print(f"Failed to save results: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
