# AletheiaReproduction

> Google DeepMind [Aletheia](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) 的独立 Python 简单复现和魔改。

## 概览

本项目实现了一个固定编排的生成-验证-修订循环：

```
Generator -> Verifier -> Reviser / Generator -> ... -> Finalizer
```

当前代码路径重点包括：

- `main.py` 提供单题 CLI 入口
- `evaluation/run_imobench.py` 提供 IMO Bench 批量评测入口
- `src/core/orchestrator.py` 负责主循环与状态持久化
- `src/tools/registry.py` 统一分发 `run_python`、`call_searcher`、`read_artifact`、`review_citation`
- `src/memory/problem_memory.py` 负责每题运行目录、`history.jsonl`、`state.json` 和工件输出

## 目录结构

```text
AletheiaReproduction/
├── main.py                  # 单题 CLI 入口
├── evaluation/
│   ├── data_loader.py       # IMO Bench 数据加载
│   └── run_imobench.py      # 批量评测入口
├── config/
│   ├── settings.yaml        # 运行配置，支持 ${ENV_VAR} 替换
│   └── prompts/             # 按 agent/stage 拆分的 prompt 文件
├── src/
│   ├── core/                # 配置、编排、终态处理
│   ├── agents/              # Generator / Verifier / Reviser
│   ├── memory/              # ProblemMemory 与运行快照
│   ├── models/              # LLM client 与传输层
│   ├── tools/               # 工具注册、搜索、代码执行、引用检查
│   └── utils/               # 解析、日志、评测等通用模块
├── runs/                    # 每题运行目录，包含 history.jsonl、state.json、artifact/
└── data/
    ├── imobench/            # IMO Bench CSV 数据集
    └── logs/                # 旧日志与样例记录
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
| `LLM_PROVIDER` | 当前使用的 provider，例如 `deepseek` 或 `volcano` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `VOLCANO_API_KEY` | Volcano Engine API Key |
| `VOLCANO_BASE_URL` | Volcano API base URL |
| `VOLCANO_MODEL` | Volcano 模型名 |
| `E2B_API_KEY` | 代码执行沙箱密钥，可选 |
| `SEMANTIC_SCHOLAR_API_KEY` | 学术检索密钥，可选 |

`config/settings.yaml` 和 `config/prompts/` 会在运行时自动加载，字符串里的 `${ENV_VAR}` 会被替换成环境变量值。

## 用法

### 单题运行

```powershell
# 从文本文件读取题目
python main.py data/problem/PB-Basic-001.txt --max-turns 3

# 直接传入题面文本
python main.py --problem "Prove that for all n>=1, n^2+n is even." --max-turns 1
```

### 批量评测

```powershell
python -m evaluation.run_imobench --dataset answerbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset proofbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset gradingbench --count 10 --max-turns 3
python -m evaluation.run_imobench --dataset all --count 10
```

### 常用输出

- 原始事件日志：`runs/{problem_id}_{timestamp}/history.jsonl`
- 状态快照：`runs/{problem_id}_{timestamp}/state.json`
- 最终答案工件：`runs/{problem_id}_{timestamp}/artifact/final_output.md`
- 可读工作日志：`runs/{problem_id}_{timestamp}/artifact/worklog.md`
- 批量评测结果：`evaluation/results/imobench_{dataset}_{timestamp}.json`

## 工具

| 工具 | 作用 |
|---|---|
| `run_python` | 在子进程沙箱中执行 Python 代码，供符号计算与验证使用 |
| `call_searcher` | 触发检索流水线并写入分层检索产物 |
| `read_artifact` | 读取 artifact markdown 的指定层 |
| `review_citation` | 检查引用路径与 claim-source 一致性 |

## 数据集

评测模块默认读取 [IMO Bench](https://imobench.github.io) 的 CSV 数据集，放置于 `data/imobench/`：

| 数据集 | 文件 |
|---|---|
| AnswerBench | `data/imobench/answerbench_v2.csv` |
| ProofBench | `data/imobench/proofbench.csv` |
| GradingBench | `data/imobench/gradingbench.csv` |

## 许可证与归属

本项目是基于 Google DeepMind [superhuman-reasoning](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) 的独立复现，遵循 Apache 2.0 许可。

详细条款请见 [LICENSE](LICENSE)。