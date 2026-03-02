"""CSV 数据加载器：读取 imobench 目录下的 Benchmark 数据集。"""

import csv
from pathlib import Path


def load_answerbench(path: str = "data/imobench/answerbench_v2.csv") -> list[dict]:
    """加载 AnswerBench 短答题数据集。

    返回 [{"problem_id": "...", "problem": "...", "answer": "..."}, ...]
    """
    return _load_csv(
        path,
        field_map={"Problem": "problem", "Short Answer": "answer"},
        id_prefix="answerbench",
    )


def load_proofbench(path: str = "data/imobench/proofbench.csv") -> list[dict]:
    """加载 ProofBench 证明题数据集。

    返回 [{"problem_id": "...", "problem": "..."}, ...]
    """
    return _load_csv(
        path,
        field_map={"Problem": "problem"},
        id_prefix="proofbench",
    )


def load_gradingbench(path: str = "data/imobench/gradingbench.csv") -> list[dict]:
    """加载 GradingBench 评分数据集。

    CSV 列映射：
      - Problem    → problem       (数学问题文本)
      - Solution   → solution      (参考解答)
      - Response   → response      (模型生成的回答，待验证)
      - Points     → points        (人工评分, 0-7)
      - Reward     → reward        (人工判定: Correct/Incorrect)
      - Problem ID → source_problem_id
      - Grading ID → grading_id
      - Problem Source → problem_source
    """
    rows = _load_csv(
        path,
        field_map={
            "Problem": "problem",
            "Solution": "solution",
            "Response": "response",
            "Points": "points",
            "Reward": "reward",
            "Problem ID": "source_problem_id",
            "Grading ID": "grading_id",
            "Problem Source": "problem_source",
        },
        id_prefix="gradingbench",
    )
    for row in rows:
        try:
            row["points"] = int(row["points"])
        except (ValueError, TypeError):
            row["points"] = 0
    return rows


def _load_csv(path: str, field_map: dict[str, str], id_prefix: str) -> list[dict]:
    """通用 CSV 加载器。

    Args:
        path: CSV 文件路径。
        field_map: CSV 列名 → 输出 key 的映射。
        id_prefix: problem_id 前缀。

    Returns:
        结构化的 dict 列表，每条含 "problem_id" 和 field_map 定义的字段。
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    results: list[dict] = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            entry: dict = {"problem_id": f"{id_prefix}_{i:04d}"}
            for csv_col, out_key in field_map.items():
                entry[out_key] = row.get(csv_col, "")
            results.append(entry)
    return results
