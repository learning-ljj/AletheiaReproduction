"""统一的选定问题评测框架 - 运行和统计整合版本

此脚本将评测执行和结果统计整合在一个文件中。支持灵活的 CSV 文件路径指定，数据集类型自动检测。

定义的三个最终状态：
  - SUCCESS: Verifier 认为成功解决（state.status == RunStatus.SUCCESS）
  - PROGRESS: 轮次耗尽但有新增引理（iteration_count >= max_turns 且有新增）
  - FAILED: 轮次耗尽，没有新增引理（iteration_count >= max_turns 但无新增）

对于非 SUCCESS 的题目，将自动调用 extract_stages.py 提取模型输出。

用法示例：
    # ProofBench 小样本（各难度 1 题，共 4 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x1.csv --max-turns 3
    
    # ProofBench 完整样本（各难度 3 题，共 12 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x3.csv --max-turns 3
    
    # AnswerBench 小样本（各领域 2 题，共 8 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x2.csv --max-turns 3
    
    # AnswerBench 大样本（各领域 10 题，共 40 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x10.csv --max-turns 3

输出：
    evaluation/results/unified_{dataset_filename}_{timestamp}.json
    evaluation/results/unified_{dataset_filename}_{timestamp}_details.csv

数据集类型自动检测：
    - 如果 CSV 包含 'Level' 列 → 识别为 ProofBench
    - 如果 CSV 包含 'Subcategory' 列 → 识别为 AnswerBench
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.core.agent import AletheiaAgent
from src.core.config import load_config, load_prompts
from src.memory.state import RunStatus
from evaluation.data_loader import lookup_ground_truth


def _configure_stdio_utf8() -> None:
    """统一 stdout/stderr 为 UTF-8，避免 Windows 重定向日志乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                continue


class UnifiedEvaluationRunner:
    """统一的评测运行器：整合运行和统计，提供三个最终状态。
    
    核心功能：
    1. 从 CSV 加载选定的问题
    2. 逐题调用 Agent 进行求解
    3. 根据以下规则确定三个最终状态之一：
       - SUCCESS: Verifier 验证成功 (state.status == RunStatus.SUCCESS)
       - PROGRESS: 轮次耗尽但有新增引理（可继续尝试）
       - FAILED: 轮次耗尽且无新增引理（Agent 已停滞）
    4. 对非 SUCCESS 题目自动调用 extract_stages.py 提取模型输出
    5. 生成 JSON 和 CSV 格式的结果文件
    
    输出文件示例：
    - JSON: evaluation/results/unified_proofbench_20260519_151523.json
    - CSV:  evaluation/results/unified_proofbench_20260519_151523_details.csv
    
    自动提取示例：
    - 提取位置: runs/{problem_id}_{timestamp}/extracted/
    - 包含文件: 0_generator.md, 0_generator_reasoning.md, ...
    
    使用示例：
        runner = UnifiedEvaluationRunner(agent, "proofbench", max_turns=3)
        result = runner.run_all()
        # 输出: JSON/CSV 结果 + 自动提取的 extracted/ 目录
    """

    def __init__(
        self,
        agent: AletheiaAgent,
        dataset_name: Literal["answerbench_v2", "proofbench"],
        csv_path: str | None = None,
        max_turns: int | None = None,
        extract_stages: bool = True,
    ):
        """初始化。

        Args:
            agent: Aletheia Agent 实例
            dataset_name: 数据集名称 ('answerbench_v2' 或 'proofbench')
            csv_path: CSV 文件路径（必须提供，或使用默认的 data/problem-case/{dataset_name}_selected.csv）
            max_turns: 单题最大轮次
            extract_stages: 是否为非 SUCCESS 题目提取阶段输出
        """
        self.agent = agent
        self.dataset_name = dataset_name
        self.max_turns = max_turns or agent.max_turns
        self.extract_stages = extract_stages

        if max_turns is not None:
            self.agent.orchestrator.max_turns = max_turns

        # 如果提供了 csv_path，直接使用；否则使用默认路径
        if csv_path:
            self.csv_path = Path(csv_path)
        else:
            self.csv_path = Path("data") / "problem-case" / f"{dataset_name}_selected.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    def load_dataset(self) -> list[dict]:
        """加载选定的问题。"""
        df = pd.read_csv(self.csv_path)

        problems = []
        for _, row in df.iterrows():
            if self.dataset_name == "answerbench_v2":
                problem_entry = {
                    "problem_id": row["Problem ID"],
                    "problem": row["Problem"],
                    "category": row.get("Category", ""),
                    "subcategory": row.get("Subcategory", ""),
                    "source": row.get("Source", ""),
                }
            elif self.dataset_name == "proofbench":
                problem_entry = {
                    "problem_id": row["Problem ID"],
                    "problem": row["Problem"],
                    "category": row.get("Category", ""),
                    "level": row.get("Level", ""),
                    "source": row.get("Source", ""),
                }
            else:
                problem_entry = dict(row)

            problems.append(problem_entry)

        return problems

    def _determine_final_status(self, state, max_turns: int) -> str:
        """根据 state 和轮次确定最终状态。

        三个最终状态的判定逻辑（按优先级）：
        1. SUCCESS: 如果 Verifier 已验证成功
        2. PROGRESS: 如果轮次已满且存在新增引理（标志：lemmas_added_last_turn == True）
        3. FAILED: 如果轮次已满但无新增引理
        4. 默认 PROGRESS: 如果轮次未满（允许继续）
        
        Args:
            state: Agent 返回的 state 对象（包含 status, iteration_count, lemmas_added_last_turn）
            max_turns: 最大轮次限制

        Returns:
            'SUCCESS', 'PROGRESS', 或 'FAILED'
        
        使用场景：
        - SUCCESS 题目无需进一步分析
        - PROGRESS 题目值得研究为什么有进展（查看 reasoning_content）
        - FAILED 题目需要找出卡点原因（查看所有提取的阶段输出）
        """
        # 首先检查 Verifier 决定
        if state.status and state.status.value == "SUCCESS":
            return "SUCCESS"

        # 如果没有成功，检查是否达到最大轮次
        if state.iteration_count >= max_turns:
            # 检查是否有新增引理
            # 这需要从 state 中判断是否在最后一个周期有新增证明
            has_new_lemma = hasattr(state, "lemmas_added_last_turn") and state.lemmas_added_last_turn
            
            if has_new_lemma:
                return "PROGRESS"
            else:
                return "FAILED"
        
        # 未达到最大轮次但也没成功
        return "FAILED"

    def _extract_stages_for_problem(self, problem_id: str, run_dir: Path) -> bool:
        """为一个问题调用 extract_stages.py 提取所有阶段的模型输出。
        
        自动调用 extract_stages.py 脚本，从 history.jsonl 中提取以下内容：
        - GENERATOR 阶段: content 和 reasoning_content（模型的推理过程）
        - REVISER 阶段: content（修正内容）
        - VERIFIER 阶段: verification（验证结果）
        
        提取的文件将保存到 run_dir/extracted/ 目录中，文件命名为：
        - {turn_id}_generator.md (Generator 的解答)
        - {turn_id}_generator_reasoning.md (Generator 的推理过程) [新增]
        - {turn_id}_reviser.md (Reviser 的修正)
        - {turn_id}_verifier.md (Verifier 的验证)

        Args:
            problem_id: 问题 ID
            run_dir: 运行目录路径（如：runs/PB-Basic-001_20260519_093502/）

        Returns:
            是否成功提取 (returncode == 0)
        
        调用示例：
            >>> self._extract_stages_for_problem("PB-Basic-001", Path("runs/PB-Basic-001_20260519_093502"))
            # 自动调用: python -m evaluation.extract_stages history.jsonl extracted/
        """
        if not self.extract_stages:
            return False

        history_file = run_dir / "history.jsonl"
        if not history_file.exists():
            return False

        try:
            # 调用 extract_stages.py
            cmd = [
                sys.executable,
                "-m",
                "evaluation.extract_stages",
                str(history_file),
                str(run_dir / "extracted"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            print(f"    [WARN] Failed to extract stages: {e}")
            return False

    def _run_single_problem(self, problem_entry: dict) -> dict:
        """运行单个问题的完整流程。
        
        流程：
        1. 从 problem_entry 提取问题 ID 和问题文本
        2. 从 data_loader 查找正确答案（ground truth）
        3. 调用 Agent.solve() 求解问题
        4. 确定最终状态（SUCCESS/PROGRESS/FAILED）
        5. 如果状态 != SUCCESS，自动调用 extract_stages 提取模型输出
        6. 返回结果字典
        
        Returns:
            {  
                'problem_id': str,
                'category': str,
                'status': 'SUCCESS' | 'PROGRESS' | 'FAILED',
                'iteration_count': int,
                'error': str | None,
                'run_dir': str | None
            }
        
        异常处理：
        - 若题目为空则返回 FAILED
        - 若 Agent 抛异常则返回 FAILED 并记录错误信息
        - 自动提取失败不会中断主流程
        """
        problem_id = problem_entry.get("problem_id", "unknown")
        problem_text = problem_entry.get("problem", "")

        if not problem_text.strip():
            return {
                "problem_id": problem_id,
                "category": problem_entry.get("category", ""),
                "status": "FAILED",
                "iteration_count": 0,
                "error": "empty problem text",
                "run_dir": None,
            }

        try:
            ground_truth, _ = lookup_ground_truth(problem_id)
            state = self.agent.solve(problem_id, problem_text, ground_truth=ground_truth)

            # 确定最终状态
            final_status = self._determine_final_status(state, self.max_turns)

            # 获取运行目录
            run_dir_path = Path("runs") / state.run_id if hasattr(state, "run_id") else None

            result = {
                "problem_id": problem_id,
                "category": problem_entry.get("category", ""),
                "status": final_status,
                "iteration_count": state.iteration_count,
                "error": None,
                "run_dir": str(run_dir_path) if run_dir_path else None,
            }

            # 对非 SUCCESS 题目提取阶段输出
            if final_status != "SUCCESS" and run_dir_path and run_dir_path.exists():
                self._extract_stages_for_problem(problem_id, run_dir_path)

            return result

        except Exception as exc:
            return {
                "problem_id": problem_id,
                "category": problem_entry.get("category", ""),
                "status": "FAILED",
                "iteration_count": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:100]}",
                "run_dir": None,
            }

    def run_all(self) -> dict:
        """运行整个数据集的所有题目并生成统计。
        
        流程：
        1. 加载 CSV 中的所有题目
        2. 逐题调用 _run_single_problem()，实时显示进度
        3. 统计三个最终状态的数量
        4. 打印实时控制台输出（每题一行）
        5. 打印最终统计摘要（成功率等）
        6. 返回完整结果对象
        
        控制台输出示例：
        ```
        [1/12] PB-Basic-001 [Algebra IMO-easy] SUCCESS (iter: 2)
        [2/12] PB-Basic-002 [Combinatorics pre-IMO] PROGRESS (iter: 3)
        ...
        Summary:
          SUCCESS: 7/12 (58.3%)
          PROGRESS: 3/12
          FAILED:   2/12
        ```

        Returns:
            {
                "dataset": str,           # "answerbench_v2" 或 "proofbench"
                "timestamp": str,         # ISO 格式时间戳
                "total": int,             # 总题数
                "success": int,           # SUCCESS 题数
                "progress": int,          # PROGRESS 题数
                "failed": int,            # FAILED 题数
                "problems": list[dict],   # 每题的详细结果
            }
        """
        dataset = self.load_dataset()
        total = len(dataset)
        results = []
        success_count = 0
        progress_count = 0
        failed_count = 0

        print(f"\n{'=' * 80}")
        print(f"Unified Evaluation: {self.dataset_name.upper()}")
        print(f"Total problems: {total}, Max turns: {self.max_turns}")
        print(f"CSV path: {self.csv_path}")
        print(f"Extract stages for non-SUCCESS: {self.extract_stages}")
        print(f"{'=' * 80}\n")

        for idx, problem_entry in enumerate(dataset, start=1):
            problem_id = problem_entry.get("problem_id", "unknown")
            category = problem_entry.get("category", "")
            level_or_subcat = (
                problem_entry.get("level") or problem_entry.get("subcategory") or ""
            )

            print(
                f"[{idx:2d}/{total}] {problem_id:30s} [{category:15s} {level_or_subcat:20s}] ",
                end="",
                flush=True,
            )

            result = self._run_single_problem(problem_entry)
            status = result.get("status")

            # 统计
            if status == "SUCCESS":
                success_count += 1
                print(f"SUCCESS (iter: {result['iteration_count']})")
            elif status == "PROGRESS":
                progress_count += 1
                print(f"PROGRESS (iter: {result['iteration_count']})")
            elif status == "FAILED":
                failed_count += 1
                error_msg = (result["error"] or "").split(":")[0][:20]
                print(f"FAILED (iter: {result['iteration_count']}, {error_msg})")
            else:
                failed_count += 1
                print(f"UNKNOWN: {status}")

            results.append(result)

        print(f"\n{'─' * 80}")
        print(f"Summary:")
        print(f"  SUCCESS: {success_count}/{total} ({100.0 * success_count / total:.1f}%)")
        print(f"  PROGRESS: {progress_count}/{total}")
        print(f"  FAILED:   {failed_count}/{total}")
        print(f"  TOTAL:    {total}")
        print(f"{'─' * 80}\n")

        return {
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "success": success_count,
            "progress": progress_count,
            "failed": failed_count,
            "problems": results,
        }


def _detect_dataset_type(csv_path: Path) -> str:
    """自动检测 CSV 文件的数据集类型 - 根据列结构判断。
    
    检测逻辑：
    - 如果包含 "Level" 列 → ProofBench (难度级别: IMO-easy, pre-IMO, 等)
    - 如果包含 "Subcategory" 列 → AnswerBench (子分类字段)
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        "proofbench" 或 "answerbench_v2"
        
    Raises:
        ValueError: 无法识别的 CSV 格式
    """
    df = pd.read_csv(csv_path, nrows=1)
    
    if "Level" in df.columns:
        return "proofbench"
    elif "Subcategory" in df.columns:
        return "answerbench_v2"
    else:
        raise ValueError(
            f"Unknown CSV format for {csv_path}. "
            "Expected either 'Level' (ProofBench) or 'Subcategory' (AnswerBench) column."
        )


def main():
    """主入口函数 - 完整使用示例和参数说明。
    
    完整使用示例：
    ```bash
    # 运行指定的 CSV 文件（自动检测类型）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x2.csv --max-turns 3
    
    # 运行 ProofBench 小样本（各难度 1 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x1.csv --max-turns 3
    
    # 运行 ProofBench 完整样本（各难度 3 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x3.csv --max-turns 3
    
    # 运行 AnswerBench V2 大样本（各领域 10 题）
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x10.csv --max-turns 3
    
    # 不自动提取非 SUCCESS 题目的输出
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x1.csv --no-extract
    
    # 指定自定义输出目录
    python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x2.csv --output-dir custom_results
    ```
    
    输出文件位置：
    - JSON 结果: {output_dir}/unified_{dataset}_{timestamp}.json
    - CSV 详情: {output_dir}/unified_{dataset}_{timestamp}_details.csv
    - 提取输出: runs/{problem_id}_{timestamp}/extracted/*.md
    
    参数说明：
    - --dataset: CSV 文件的相对路径 (如 data/problem-case/answerbench_v2_selected_4x2.csv)
               数据集类型自动从 CSV 列结构检测（Level→ProofBench, Subcategory→AnswerBench）
    - --max-turns: 每题最大轮次 (默认从配置读取)
    - --output-dir: 结果输出目录 (默认: evaluation/results)
    - --no-extract: 不提取非 SUCCESS 题目的模型输出
    
    流程说明：
    1. 接受 CSV 文件路径，自动检测数据集类型
    2. 为该数据集加载 CSV 中的所有题目
    3. 逐题调用 Agent 求解，显示实时进度
    4. 对非 SUCCESS 题目自动调用 extract_stages.py 提取模型输出
    5. 生成 JSON 和 CSV 格式的结果文件
    6. 打印最终统计摘要（成功率、各状态题数等）
    """
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for selected problems"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset CSV file (e.g., data/problem-case/answerbench_v2_selected_4x2.csv). "
             "Dataset type is auto-detected from CSV structure.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns per problem (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Output directory for results JSON/CSV (default: evaluation/results)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract stages for non-SUCCESS problems",
    )

    args = parser.parse_args()

    _configure_stdio_utf8()

    # 验证并规范化数据集路径
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    # 自动检测数据集类型
    try:
        detected_dataset_type = _detect_dataset_type(dataset_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    prompts = load_prompts()
    agent = AletheiaAgent(config=config, prompts=prompts)

    try:
        runner = UnifiedEvaluationRunner(
            agent,
            detected_dataset_type,
            csv_path=str(dataset_path),
            max_turns=args.max_turns,
            extract_stages=not args.no_extract,
        )
        result = runner.run_all()

        # 保存 JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 从文件名生成更有意义的输出名称
        dataset_file_stem = dataset_path.stem  # e.g., "answerbench_v2_selected_4x2"
        json_output = output_dir / f"unified_{dataset_file_stem}_{timestamp}.json"
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"JSON saved: {json_output}\n")

        # 保存 CSV
        csv_output = (
            output_dir / f"unified_{dataset_file_stem}_{timestamp}_details.csv"
        )
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "problem_id",
                    "category",
                    "status",
                    "iteration_count",
                    "error",
                    "run_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(result["problems"])
        print(f"CSV saved: {csv_output}\n")

        # 最终统计
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        total = result["total"]
        success = result["success"]
        progress = result["progress"]
        failed = result["failed"]
        print(
            f"\n{detected_dataset_type.upper()}:"
            f"\n  SUCCESS: {success}/{total} ({100.0 * success / total:.1f}%)"
            f"\n  PROGRESS: {progress}/{total}"
            f"\n  FAILED:   {failed}/{total}"
        )
        print(f"{'=' * 80}\n")

    except Exception as exc:
        print(f"Error running evaluation: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
