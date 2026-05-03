# 架构与设计文档

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        命令行入口 (main.py)                      │
│  python main.py <problem_file> [--max-turns N]                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    参数解析与配置加载                             │
│  build_parser() → load_config() / load_prompts()                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AletheiaAgent                             │
│  组装 LLMClient、工具注册表、AgentPipeline、Orchestrator         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                             │
│  初始化 ProblemMemory → 运行主循环 → 保存状态与工件              │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Generator   │ │   Verifier   │ │   Reviser    │
│  初始解答    │ │  判定与工具   │ │  局部修订    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           ToolExecutor                           │
│  run_python / call_searcher / read_artifact / review_citation   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ProblemMemory / Finalizer                   │
│  history.jsonl / state.json / manifest.json / final_output.md    │
└─────────────────────────────────────────────────────────────────┘
```

## 类设计

### AletheiaAgent

`AletheiaAgent` 是高层门面，负责把配置、prompt、LLM client、工具和编排器装配成可执行的推理系统。

### AgentPipeline

`AgentPipeline` 负责按阶段创建 `GeneratorAgent`、`VerifierAgent` 和 `ReviserAgent`，并为不同阶段注入工具白名单。

### Orchestrator

`Orchestrator` 是控制流中心，负责：

- 创建并绑定 `ProblemMemory`
- 执行生成、验证、修订循环
- 记录阶段事件和告警
- 将终态交给 `FinalizerEngine`

### FinalizerEngine

`FinalizerEngine` 将最终状态转化为可读输出，并持久化：

- 最终事件 `FINAL`
- `state.json`
- `artifact/final_output.md`
- `manifest.json`

### ProblemMemory

`ProblemMemory` 保存单题运行期间的持久化数据，是 `history.jsonl`、状态快照和 artifact 的唯一写入点。

## 数据流

### 单题处理流程

```
题目文本
    ↓
main.py 读取参数与配置
    ↓
创建 AletheiaAgent
    ↓
Orchestrator 初始化 ProblemMemory
    ↓
Generator 产出初稿
    ↓
Verifier 执行三阶段检查
    ↓
决策分支:
  CORRECT       -> Finalizer 生成最终答案
  MINOR_FLAW    -> Reviser 修订后重新验证
  CRITICAL_FLAW -> 回到 Generator 重新生成
    ↓
Finalizer 写入 state.json / history.jsonl / artifact
```

### 验证器逻辑

Verifier 当前的职责是：

- 读取候选证明文本
- 触发工具调用，主要包括 `run_python`、`read_artifact` 和 `review_citation`
- 解析 XML 契约中的 `verdict`、`verification`、`verified_lemmas` 和 `citation_review`
- 将 `CORRECT`、`MINOR_FLAW`、`CRITICAL_FLAW` 三路决策返回给编排器

## 配置参数流

```
config/settings.yaml
    │
    ├─ provider / deepseek / volcano
    ├─ agent.max_turns
    ├─ agent.worklog_llm_timeout_seconds
    ├─ retrieval.sources / retry / dedup
    ├─ verifier.citation_review
    └─ repro.snapshot_prompts / snapshot_settings
```

`load_config()` 会自动将 `${ENV_VAR}` 占位符替换为运行时环境变量。

## 输出文件结构

```
runs/
└── {problem_id}_{timestamp}/
    ├── history.jsonl
    ├── state.json
    ├── manifest.json
    └── artifact/
        ├── final_output.md
        └── worklog.md

evaluation/
└── results/
    └── imobench_{dataset}_{timestamp}.json
```

## 关键设计决策

### 1. 为什么使用固定编排器

固定循环更容易复现、调试和做回归测试。相比动态工作流图，这里把控制权集中在 `Orchestrator`，可以更稳定地对齐 Aletheia 的核心行为。

### 2. 为什么按阶段限制工具

不同阶段只暴露必要工具，能减少提示词污染和误调用：

- Generator / Reviser 侧重读取已有工件
- Verifier 允许运行代码、读取工件并做引用检查

### 3. 为什么持久化单题运行目录

按题目和时间戳分目录可以避免日志覆盖，并让 `history.jsonl`、`state.json`、`worklog.md` 形成可追溯的一组产物，便于离线分析。

## 错误处理策略

```
异常发生
    ↓
工具层 / Agent 层 / 编排层
    ↓
结构化错误包或直接抛出
    ↓
Orchestrator 记录到 history.jsonl 和 state.json
    ↓
main.py / evaluation/run_imobench.py 决定是否返回非零退出码
```

核心原则是：

- 可恢复问题尽量结构化返回
- 不可恢复问题直接中止当前题目
- 批量评测遇到单题错误时继续下一题