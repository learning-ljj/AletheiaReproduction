# AletheiaReproduction

> Google DeepMind [Aletheia](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) 的独立 Python 复现，面向本地实验、数学推理调试与批量评测。

## 概览

本项目实现了一个固定编排的生成-验证-修订循环：

```
Generator -> Verifier -> Reviser / Generator -> ... -> Finalizer
```

支持 **两条推理分支**：

| 分支 | 说明 | Prompt 目录 | 批量评测入口 |
|---|---|---|---|
| **数学推理 (math)** | 面向 IMO 级别数学证明的推理循环，含检索增强 | `config/prompts/` | `evaluation/run_imobench.py` |
| **通用推理 (general)** | 面向通用问答（常识、科学、逻辑等）的轻量化推理，不含检索 | `config/prompts_general/` | `evaluation/run_general365.py` |

通过 `main.py` 的 `--task {math,general}` 参数切换分支。

当前代码路径重点包括：

- `main.py` 提供单题 CLI 入口，支持 `--task` 切换推理分支
- `evaluation/run_general365.py` 提供 General365 通用推理批量评测入口
- `evaluation/run_imobench.py` 提供 IMO Bench 数学推理批量评测入口
- `evaluation/run_and_analyze_selected.py` 提供 ProofBench / AnswerBench 选定题统一评测入口
- `src/core/orchestrator.py` 负责主循环与状态持久化
- `src/tools/registry.py` 统一分发 `run_python`、`call_searcher`、`read_artifact`、`review_citation`
- `src/memory/problem_memory.py` 负责每题运行目录、`history.jsonl`、`state.json` 和工件输出

## 目录结构

```text
AletheiaReproduction/
├── main.py                       # 单题 CLI 入口（--task 切换推理分支）
├── evaluation/
│   ├── data_loader.py            # IMO Bench / General365 数据加载
│   ├── run_imobench.py           # IMO Bench 批量评测入口
│   ├── run_general365.py         # General365 通用推理批量评测入口
│   ├── run_and_analyze_selected.py # ProofBench / AnswerBench 选定题统一评测
│   └── extract_stages.py         # 从 history.jsonl 提取阶段输出（含 LaTeX 规范化）
├── config/
│   ├── settings.yaml             # 运行配置，支持 ${ENV_VAR} 替换
│   ├── prompts/                  # 数学推理 prompt 模板（按 agent/stage 拆分）
│   └── prompts_general/          # 通用推理 prompt 模板（generator / reviser / verifier / final）
├── src/
│   ├── core/                     # 配置、编排、终态处理
│   ├── agents/                   # Generator / Verifier / Reviser
│   ├── memory/                   # ProblemMemory 与运行快照
│   ├── models/                   # LLM client 与传输层
│   ├── tools/                    # 工具注册、搜索、代码执行、引用检查
│   └── utils/                    # 解析、日志、评测等通用模块
├── data/
│   ├── imobench/                 # IMO Bench CSV 数据集
│   └── problem-case/             # 精选题目（CSV + TXT），含 General365 / ProofBench / AnswerBench
└── runs/                         # 运行产物（history.jsonl / state.json / artifact）
```

更多架构说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 安装

### 依赖

- Python 3.13+
- `uv`（推荐）或 `pip`

### 安装步骤

```powershell
git clone https://github.com/learning-ljj/AletheiaReproduction.git
cd AletheiaReproduction

uv pip install -r requirements.txt
# 或
pip install -r requirements.txt
```

### 环境变量

复制 `.env` 并配置模型提供方相关环境变量：

| 变量 | 说明 |
|---|---|
| `LLM_PROVIDER` | 当前使用的 provider，例如 `deepseek` / `volcano` / `litellm` |
| `OPENAI_API_KEY` | LiteLLM/OpenAI 兼容 API Key |
| `OPENAI_BASE_URL` | LiteLLM/OpenAI 兼容 Base URL |
| `LITELLM_MODEL` | LiteLLM 模型名（例如 `gpt-5.5`） |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `VOLCANO_API_KEY` | Volcano Engine API Key |
| `VOLCANO_BASE_URL` | Volcano API base URL |
| `VOLCANO_MODEL` | Volcano 模型ID |
| `E2B_API_KEY` | 代码执行沙箱密钥，可选 |
| `SEMANTIC_SCHOLAR_API_KEY` | 学术检索密钥，可选 |

`config/settings.yaml` 和 `config/prompts/` 会在运行时自动加载，字符串里的 `${ENV_VAR}` 会被替换成环境变量值。

## 用法

### 单题运行（main.py）

`main.py` 是统一的单题入口，通过 `--task` 参数切换推理分支。

```powershell
# 数学推理（默认）：从文本文件读取题目
python main.py data/problem-case/PB-Basic-001.txt --max-turns 3

# 数学推理：直接传入题面文本
python main.py --problem "Prove that for all n>=1, n^2+n is even." --max-turns 1

# 通用推理：从文本文件读取题目
python main.py data/problem-case/general365-253-2.txt --task general --max-turns 3

# 通用推理：按 problem_id 自动查找 CSV 中的 ground truth
python main.py data/problem-case/general365_selected_20.csv --task general --max-turns 3
```

#### main.py CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `problem` (位置参数) | str | — | 题目文件路径或 CSV 文件路径 |
| `--problem` | str | — | 直接传入题面文本（与位置参数二选一） |
| `--task` | str | `math` | 推理分支：`math`（数学推理）或 `general`（通用推理） |
| `--max-turns` | int | `3` | 最大精炼轮次 |

### 批量评测

#### General365 通用推理评测

```powershell
# 运行全部 20 题
python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --max-turns 3

# 仅运行前 5 题
python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --limit 5

# 指定输出目录，不提取阶段输出
python -m evaluation.run_general365 --dataset data/problem-case/general365_selected_20.csv --output-dir my_results --no-extract
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dataset` | str | `data/problem-case/general365_selected_20.csv` | General365 CSV 数据集路径 |
| `--max-turns` | int | `3` | 每个问题的最大精炼轮次 |
| `--limit` | int | `None` | 仅运行前 N 道题（默认全部运行） |
| `--output-dir` | str | `results/general365` | 结果 JSON 输出目录 |
| `--no-extract` | flag | `False` | 跳过对非 SUCCESS 题目的阶段输出提取 |

最终状态：`SUCCESS` / `PROGRESS` / `FAILED`。非 SUCCESS 题目会自动调用 `extract_stages.py` 提取各阶段模型输出（含 LaTeX 数学公式规范化 `\[...\]` → `$$...$$`）。

#### IMO Bench 批量评测

```powershell
python -m evaluation.run_imobench --dataset answerbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset proofbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset gradingbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset all --count 10
```

#### ProofBench / AnswerBench 选定题统一评测

```powershell
# ProofBench 小样本（各难度 1 题 × 4）
python -m evaluation.run_and_analyze_selected --dataset data/problem-case/proofbench_selected_4x1.csv --max-turns 3

# AnswerBench 大样本（各领域 10 题 × 4）
python -m evaluation.run_and_analyze_selected --dataset data/problem-case/answerbench_v2_selected_4x10.csv --max-turns 3
```

### 常用输出

- 原始事件日志：`runs/{problem_id}_{timestamp}/history.jsonl`
- 状态快照：`runs/{problem_id}_{timestamp}/state.json`
- 最终答案工件：`runs/{problem_id}_{timestamp}/artifact/final_output.md`
- 可读工作日志：`runs/{problem_id}_{timestamp}/artifact/worklog.md`
- 阶段提取输出（非 SUCCESS 题目）：`runs/{problem_id}_{timestamp}/extracted/{turn_id}_generator.md` 等
- 批量评测结果：
  - IMO Bench：`evaluation/results/imobench_{dataset}_{timestamp}.json`
  - General365：`results/general365/general365_{dataset_stem}_{timestamp}.json`
  - 选定题统一评测：`evaluation/results/unified_{dataset_filename}_{timestamp}.json`

## 工具

| 工具 | 作用 |
|---|---|
| `run_python` | 在子进程沙箱中执行 Python 代码，供符号计算与验证使用 |
| `call_searcher` | 触发检索流水线并写入分层检索产物 |
| `read_artifact` | 读取 artifact markdown 的指定层 |
| `review_citation` | 检查引用路径与 claim-source 一致性 |

## 数据集

### IMO Bench（数学推理）

评测模块默认读取 [IMO Bench](https://imobench.github.io) 的 CSV 数据集，放置于 `data/imobench/`：

| 数据集 | 文件 |
|---|---|
| AnswerBench | `data/imobench/answerbench_v2.csv` |
| ProofBench | `data/imobench/proofbench.csv` |
| GradingBench | `data/imobench/gradingbench.csv` |

### General365（通用推理）

通用推理数据集 `data/problem-case/general365_selected_20.csv`，包含从 [General365](https://github.com/GeneralReasoning/General365) 选择的 20 道题，覆盖 7 个类别：

| 列名 | 说明 |
|---|---|
| `Problem ID` | 问题唯一标识（如 `general365-1`） |
| `Problem` | 问题文本 |
| `Short Answer` | 标准答案（ground truth） |
| `Category` | 大类 |
| `Subcategory` | 答案类型 |
| `Source` | 来源 |

### 精选题目（problem-case）

`data/problem-case/` 目录包含用于快速验证的精选题目子集：

| 文件 | 用途 |
|---|---|
| `general365_selected_20.csv` | General365 精选 20 题（通用推理） |
| `proofbench_selected_4x1.csv` | ProofBench 小样本（各难度 1 题 × 4） |
| `proofbench_selected_4x3.csv` | ProofBench 中等样本（各难度 3 题 × 4） |
| `answerbench_v2_selected_4x2.csv` | AnswerBench 小样本（各领域 2 题 × 4） |
| `answerbench_v2_selected_4x10.csv` | AnswerBench 大样本（各领域 10 题 × 4） |

## 许可证与归属

本项目是基于 Google DeepMind [superhuman-reasoning](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) 的独立复现，遵循 Apache 2.0 许可。

详细条款请见 [LICENSE](LICENSE)。