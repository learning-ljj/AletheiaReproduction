"""CSV 数据加载器：读取 imobench 目录下的 Benchmark 数据集。"""

import csv
import re
from pathlib import Path


def load_answerbench_full(path: str = "data/imobench/answerbench_v2.csv") -> list[dict]:
    """加载 answerbench_v2 完整字段（含 Category、Subcategory、Source）。"""
    return _load_csv(
        path,
        field_map={
            "Problem": "problem",
            "Short Answer": "answer",
            "Category": "category",
            "Subcategory": "subcategory",
            "Source": "source",
        },
        id_prefix="answerbench",
        problem_id_column="Problem ID",
    )


def load_proofbench_full(path: str = "data/imobench/proofbench.csv") -> list[dict]:
    """加载 proofbench 完整字段（含 Solution、Category、Level、Source）。"""
    return _load_csv(
        path,
        field_map={
            "Problem": "problem",
            "Solution": "solution",
            "Category": "category",
            "Level": "level",
            "Source": "source",
        },
        id_prefix="proofbench",
        problem_id_column="Problem ID",
    )


def load_gradingbench_full(path: str = "data/imobench/gradingbench.csv") -> list[dict]:
    """加载 gradingbench 完整字段（含 Grading guidelines）。"""
    rows = _load_csv(
        path,
        field_map={
            "Problem": "problem",
            "Solution": "solution",
            "Grading guidelines": "grading_guidelines",
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
        row["points"] = _safe_int(row.get("points"))
    return rows


def lookup_ground_truth(problem_id: str) -> tuple[str | None, str | None]:
    """按 problem_id 自动回填 ground_truth。

    Returns:
        (ground_truth, source_hint)
    """
    normalized = _normalize_problem_id(problem_id)
    if not normalized:
        return None, None

    # 优先匹配 answerbench 短答。
    try:
        for row in load_answerbench_full():
            if (row.get("problem_id") or "").strip() == normalized:
                ans = (row.get("answer") or "").strip()
                if ans:
                    return ans, "answerbench_v2.csv:Short Answer"
    except Exception:
        pass

    # 兼容 PB-* 等题号：从 gradingbench 的 source_problem_id 回填参考解答。
    try:
        for row in load_gradingbench_full():
            if (row.get("source_problem_id") or "").strip() == normalized:
                sol = (row.get("solution") or "").strip()
                if sol:
                    return sol, "gradingbench.csv:Solution"
    except Exception:
        pass

    return None, None


def _load_csv(
    path: str,
    field_map: dict[str, str],
    id_prefix: str,
    problem_id_column: str | None = None,
) -> list[dict]:
    """通用 CSV 加载器。

    Args:
        path: CSV 文件路径。
        field_map: CSV 列名 → 输出 key 的映射。
        id_prefix: problem_id 前缀。
        problem_id_column: 若存在则优先使用该列作为 problem_id。

    Returns:
        结构化的 dict 列表，每条含 "problem_id" 和 field_map 定义的字段。
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    results: list[dict] = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        required = set(field_map.keys())
        if problem_id_column:
            required.add(problem_id_column)
        missing = sorted(required - header)
        if missing:
            raise ValueError(
                f"CSV schema mismatch for {path}: missing columns {missing}. "
                f"Available columns: {sorted(header)}"
            )
        for i, row in enumerate(reader):
            problem_id = row.get(problem_id_column, "").strip() if problem_id_column else ""
            entry: dict = {"problem_id": problem_id or f"{id_prefix}_{i:04d}"}
            for csv_col, out_key in field_map.items():
                entry[out_key] = row.get(csv_col, "")
            results.append(entry)
    return results


def _normalize_problem_id(problem_id: str | None) -> str:
    if not problem_id:
        return ""
    x = problem_id.strip()
    x = re.sub(r"_\d{8}_\d{6}$", "", x)
    x = re.sub(r"\(\d+\)$", "", x)
    return x


def _safe_int(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        return 0
