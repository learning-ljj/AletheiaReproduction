# 评测模块 (Evaluation Module)

## 模块结构

```
evaluation/
  ├── __init__.py              # 包初始化
  ├── data_loader.py           # 数据集加载器
  ├── run_imobench.py          # 批量评测脚本
  ├── results/                 # 评测结果输出目录
```

## 新脚本：run_imobench.py

### 功能

批量运行四个 IMOBench 数据集，记录成功/失败的题数统计。

- **支持的数据集**：answerbench、proofbench、gradingbench
- **输出**：控制台统计表格 + JSON 结果文件
- **评测指标**：SUCCESS（成功）/ FAILURE（失败）/ TOTAL（总数）

### 设计原则

1. **高内聚低耦合**
   - `BenchmarkRunner` 类职责单一，只负责单个数据集的运行
   - 通过依赖注入（Agent）解耦

2. **不做兜底性处理，错误直接暴露**
   - 单题运行失败时返回 ERROR 状态，允许调用方处理
   - 数据集加载、Agent 初始化等异常直接抛出

3. **简洁的中文注释和 docstring**
   - 每个类/方法都有中文说明
   - 参数、返回值类型明确

### 使用方法

#### 1. 运行全部数据集

```bash
python -m evaluation.run_imobench
```

#### 2. 运行单个数据集

```bash
# 只评测 answerbench
python -m evaluation.run_imobench --dataset answerbench

# 只评测 proofbench
python -m evaluation.run_imobench --dataset proofbench

# 只评测 gradingbench
python -m evaluation.run_imobench --dataset gradingbench
```

#### 3. 限制题数和轮次

```bash
# 评测前10题，最多5轮
python -m evaluation.run_imobench --dataset all --count 10 --max-turns 5

# 只评测 answerbench 的前20题
python -m evaluation.run_imobench --dataset answerbench --count 20
```

### 输出示例

#### 控制台输出

```
======================================================================
Benchmark: ANSWERBENCH
Total problems: 30, Max turns: 3
======================================================================

[  1/30] answerbench_0000                         ✓ SUCCESS
[  2/30] answerbench_0001                         ✗ PROGRESS
[  3/30] answerbench_0002                         ⊘ SKIP (empty problem text)
...
──────────────────────────────────────────────────────────────────────
Summary: SUCCESS=15, FAILURE=15, TOTAL=30
Success rate: 50.0%
──────────────────────────────────────────────────────────────────────
```

#### JSON 结果文件

`evaluation/results/imobench_answerbench_20260501_120000.json`

```json
{
  "dataset": "answerbench",
  "total": 30,
  "success": 15,
  "failure": 15,
  "problems": [
    {
      "problem_id": "answerbench_0000",
      "status": "SUCCESS",
      "iteration_count": 2,
      "error": null
    },
    {
      "problem_id": "answerbench_0001",
      "status": "PROGRESS",
      "iteration_count": 3,
      "error": null
    },
    ...
  ]
}
```

### 核心类和方法

#### `BenchmarkRunner` 类

| 方法 | 说明 |
|------|------|
| `__init__()` | 初始化运行器，设置数据集、最大题数、轮次 |
| `load_dataset()` | 加载指定 benchmark 数据集 |
| `_run_single_problem()` | 运行单个问题，返回结果字典 |
| `run_all()` | 运行整个 benchmark 并返回统计结果 |

#### `run_benchmark()` 函数

简化的运行接口，返回统计结果字典。

#### `save_results()` 函数

将结果保存到 `evaluation/results/` 目录的 JSON 文件。

### 状态说明

| 状态 | 含义 | 计数 |
|------|------|------|
| `SUCCESS` | Aletheia 判定成功 | 成功数 |
| `PROGRESS` | 有进展但未完全解决 | 失败数 |
| `FAILED` | 失败 | 失败数 |
| `SKIP` | 跳过（如空问题） | 失败数 |
| `ERROR` | 运行异常 | 失败数 |

### 扩展建议

如需添加更多的评测指标或修改统计方式：

1. 继承 `BenchmarkRunner` 类
2. 重写 `_run_single_problem()` 方法以添加自定义逻辑
3. 修改 `run_all()` 中的统计部分

例子：

```python
class CustomRunner(BenchmarkRunner):
    def _run_single_problem(self, problem_entry: dict) -> dict:
        result = super()._run_single_problem(problem_entry)
        # 添加自定义评分逻辑
        return result
```

---

**维护者**: Aletheia 项目  
**最后更新**: 2026-05-01  
**状态**: ✅ 生产环境
