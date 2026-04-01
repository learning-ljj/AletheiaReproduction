# ResearchMathAgent 架构重构与 MVP 设计文档

## 0. 文档目的

这份文档做两件事：

1. 先用大白话把你要做的重构讲清楚（做什么、为什么做、先后顺序是什么）。
2. 再给出一个可以直接落地的 MVP 架构方案（目录、模块、数据流、配置、依赖、代码改造点）。

目标不是一步到位做成“完美多智能体”，而是先把你当前的单向流水线，重构到一个可运行、可扩展、可复现的最小可行版本。

---

## 1. 先讲人话：这次重构到底要做什么

你现在的系统本质是：

- 一个调度器按固定顺序调用函数。
- 只有 Verifier 能用工具。
- 所有状态都堆在一个 history 里，属于“流水账”。

你要变成的是：

- 三个有状态的 Agent（Generator/Reviser/Verifier），每个阶段有自己的短期记忆。
- 每个问题有一个 ProblemMemory，负责公共记忆与落盘。
- 所有 Agent 都能调用不同的工具，Generator 还可以拉起 Searcher 子 Agent。
- 引理和论文都分层存储，按需读取，不把长文本一次性塞给模型。

一句话总结：
从“函数流水线”升级为“Agent 协作 + 问题级记忆 + 分层知识加载”的可扩展框架。

---

## 2. 核心模块（大白话版）

### 2.1 Orchestrator（总调度）

作用：

- 决定这一步该叫谁干活（Generator、Verifier、Reviser）。
- 解析 Verifier 输出里的标签（尤其是 verified_lemmas）。
- 把中间结果写进 ProblemMemory。

它像“项目经理”，不做数学推理，只做流程与状态推进。

### 2.2 Generator Agent（出方案的人）

作用：

- 先产出结构化 solution。
- 必须把候选引理写在 solution 前面，且用 lemma 标签块标出来。
- 如果引用外部知识，必须写 cite:文件路径。

它不是“只会胡思乱想的函数”，而是可多轮工具调用的 Agent。

### 2.3 Verifier Agent（审稿人）

作用：

- 验证整份解答，并优先审查引理。
- 引理没有完整证明就拒绝通过。
- 把验证通过的引理放进 verified_lemmas 标签，交给 Orchestrator 入库。
- 给出路由级别的结论（CORRECT/MINOR_FLAW/CRITICAL_FLAW）。

它是质量闸门，不负责替 Generator 补证明。

### 2.4 Reviser Agent（修订人）

作用：

- 只根据 verifier report 修补当前解答。
- 不重写整份答案，尽量保留正确部分。

它是“补洞”角色，不是“重开一题”。

### 2.5 ProblemMemory（每题记忆中枢）

作用：

- 保存当前题目的 ProofState 关键字段。
- 维护历史事件 history.jsonl。
- 管理 artifact 目录：lemmas、papers、errors、citations。
- 提供分层读取接口（只读第一层摘要、第二层正文、第三层来源）。

它是每题一个实例，不是全局跨题单例。

### 2.6 Searcher 子 Agent（文献助手）

作用：

- 被 Generator 按需调用。
- 检索和清洗论文，按三层结构写入 papers 目录。
- 做去重（避免重复抓同一篇）。

它是“工具形态的子 Agent”，生命周期由 Generator 单次阶段触发。

---

## 3. 核心流程（一步步）

1. Orchestrator 接到题目，创建 ProblemMemory(problem_id)。
2. ProblemMemory 初始化目录、state.json、history.jsonl。
3. Orchestrator 从 ProblemMemory 拉取摘要上下文（引理摘要、论文摘要、历史错误摘要），喂给 Generator。
4. Generator 运行 ReAct 循环，必要时调用 Searcher 或读取分层文件，输出 solution（含多个 lemma 块）。
5. Orchestrator 记录事件到 history.jsonl。
6. Verifier 对整份 solution 做验证，同时重点处理 lemma：
   - 有完整证明且通过：写入 verified_lemmas。
   - 无完整证明或失败：进入 error report。
7. Orchestrator 解析 verified_lemmas，调用 ProblemMemory.add_lemma 落盘。
8. 根据 Verifier 裁决路由：
   - CORRECT 结束。
   - MINOR_FLAW 交 Reviser。
   - CRITICAL_FLAW 交 Generator。
9. 达到 max_turns 后走 Final 模块，生成最终输出与 References。

---

## 4. 显式需求与隐式需求拆解

### 4.1 显式需求（你已经明确提出）

- ProofState 升级为 ProblemMemory 体系。
- 三个节点函数重构成 Agent 对象，具备阶段内短期记忆。
- Generator 输出中必须有 lemma 标签；Verifier 输出 verified_lemmas。
- artifact 分层存储，ProblemMemory 只持有摘要。
- Searcher 作为子 Agent，由 Generator 调用并去重。
- 引用追踪要求 cite:文件路径，最终输出 References 与 BibTeX。

### 4.2 隐式需求（不做会踩坑）

- 需要统一标签协议，否则 parser 会频繁报错。
- 需要原子写文件与目录隔离，否则并发跑实验会互相污染。
- 需要工具调用轮次上限，否则 Agent 可能陷入无限工具循环。
- 需要从“日志可读”升级到“日志可重放”，否则实验不可复现。
- 需要把“模型输出格式失败”当作常态处理（重试与降级），不是异常处理。

---

## 5. 需求冲突裁决（必须先定规则）

### 5.1 ProblemMemory 是单例还是每题一个实例

裁决：每题一个实例。

解释：

- 运行时“所有 Agent 共享”说的是同一题内共享同一个 ProblemMemory 实例。
- 不同题必须隔离目录和状态，避免污染。

### 5.2 history.jsonl 在哪里

裁决：放在 {problem_id}/history.jsonl，与 artifact 同级。

解释：

- 这是你给的目录结构中最一致、最清晰、最易维护的方案。
- 不再使用旧 data/logs/{problem_id}.jsonl 作为主链路。

---

## 6. MVP 架构设计（正式版）

## 6.1 核心设计原则

- 单一职责：调度、推理、验证、记忆、工具分离。
- 问题隔离：每题独立目录。
- 分层暴露：摘要先注入，正文按需读，引用最后展开。
- 可复现：状态、历史、配置、依赖版本全部可追踪。
- 先稳后强：先做 MVP，可运行优先于花哨能力。

## 6.2 模块划分（按你要求的类别）

### 数据加载

- 题目数据与基准数据加载。
- 输入标准化（problem_id、problem_text、ground_truth）。

### 预处理

- Prompt 组装。
- ProblemMemory 摘要注入。
- 标签契约预检查。

### 核心逻辑处理

- Orchestrator 路由。
- Agent 阶段循环。
- Lemma 验证写入。

### 模型接口

- LLMClient 封装（thinking、tools、streaming、重试）。

### 推理管道

- Generator/Reviser/Verifier Agent 的 run。
- Searcher 子 Agent 调用桥接。

### 结果解析

- 结构化标签解析：solution、lemma、verdict、verified_lemmas、cite。

### 自动化评估

- answerbench/proofbench/gradingbench 脚本。
- 可追加 lemma 通过率、引用覆盖率等指标。

### 日志与监控

- history.jsonl 原始事件。
- state.json 快照。
- artifact 实体内容。

### 配置管理

- settings.yaml + prompts.yaml。
- 新增 memory 与 agent 工具配置块。

### 测试

- 单测：Parser/Memory/Tool。
- 集成测试：一题完整流程。
- 回归测试：固定种子与固定 prompt 的可复现输出。

---

## 7. 目标目录结构树（MVP，精确到 py 与关键配置）

说明：下面是建议的重构后目录。标注“保留”的文件为现有文件延续；“新增”是建议添加。

AletheiaReproduction/
  main.py                               # 保留：CLI 入口，创建 ProblemMemory 并启动 Orchestrator
  architecture.md                       # 新增：本架构文档
  requirements.txt                      # 保留：依赖锁定
  README.md                             # 保留：使用说明

  config/
    settings.yaml                       # 保留并扩展：provider、timeouts、agent与memory配置
    prompts/                            # 新增：存放各agent的prompt配置
      generator.yaml                    # 新增：Generator的prompt
      reviser.yaml                      # 新增：Reviser的prompt
      verifier.yaml                     # 新增：Verifier的prompt
      searcher.yaml                     # 新增：Searcher的prompt

  states/                               # 新增：状态层，与src同级 (修改文件夹名以防与state.py重复)
    __init__.py                         # 新增
    state.py                            # 保留并迁移：ProofState 仅保留关键运行字段
    problem_memory.py                   # 新增：每题记忆中枢与 artifact 读写

  scripts/
    run_imobench.py                     # 保留：批评测
    run_proofbench_advanced.py          # 保留：proofbench 高难子集

  src/
    __init__.py                         # 保留

    core/
      __init__.py                       # 保留
      agent.py                          # 废弃或仅作简单入口，将原有逻辑转移至 Orchestrator 或直接实例化
      orchestrator.py                   # 保留重构：中心调度、解析 verified_lemmas、写 history
      finalizer.py                      # 保留增强：引用展开与 References 输出
      config.py                         # 保留：配置加载
      contracts.py                      # 新增：标签与 schema 常量

    agents/
      __init__.py                       # 新增
      base.py                     # 新增：通用 Agent 基类（messages、tool loop、reset）
      generator.py                # 新增：生成解答与 lemma
      reviser.py                  # 新增：修订解答与 lemma
      verifier.py                 # 新增：验证解答与 lemma
      searcher.py                 # 新增：检索、论文清洗、内容提取与检索去重

    models/
      __init__.py                       # 保留
      llm_client.py                     # 保留：统一 chat 与 tool calling

    tools/
      __init__.py                       # 保留
      registry.py                       # 保留重构：按 Agent 分别注册工具集合与 max_tool_rounds
      code_executor.py                  # 保留
      artifact_reader.py                # 新增：按层读取 markdown 第一/二/三层
      searcher_bridge.py                # 新增：Generator 调用 Searcher 的工具桥接

    utils/
      __init__.py                       # 保留
      logging/                          # 新增：日志与事件记录相关
        __init__.py
        logger.py                       # 保留但迁移：ProblemMemory 体系下统一事件写入
        raw_log_reader.py               # 保留
        worklog_builder.py              # 保留
      parsing/                          # 新增：解析器与处理相关
        __init__.py
        parser.py                       # 保留增强：新增 lemma/verified_lemmas/cite 解析
        markdown_layer.py               # 新增：分层 markdown 读写工具
        reference_builder.py            # 新增：把 cite:路径 转成标准引用与 bib
      evaluation/                       # 新增：测试数据加载与评估
        __init__.py
        data_loader.py                  # 保留
        evaluator.py                    # 保留

  tests/
    realtime_e2e_monitor.py             # 保留
    test_problem_memory.py              # 新增：状态/落盘/分层读取测试
    test_parser_contracts.py            # 新增：标签契约测试
    test_orchestrator_routing.py        # 新增：路由与写入行为测试
    test_searcher_dedup.py              # 新增：去重行为测试

  runs/
    {problem_id}/                       # 新增：每题隔离目录（运行产物）
      history.jsonl                     # 原始事件日志，由 orchestrator 写
      state.json                        # ProofState 快照
      artifact/
        lemmas/
          001.md                        # 验证通过引理
          002.md
        papers/
          arXiv_2501.12345.md           # 清洗后论文三层文档
        errors/
          001.md                        # 错误分析报告
        citations.bib                   # 引用导出（可选）

---

## 8. 状态与上下文管理（存储位置、流转方式、示例）

## 8.1 存储在哪里

- 运行内存：
  - Agent.messages（阶段内短期记忆，阶段结束清空）
  - Orchestrator 当前轮状态（临时）

- 文件持久化：
  - runs/{problem_id}/state.json
  - runs/{problem_id}/history.jsonl
  - runs/{problem_id}/artifact/lemmas/*.md
  - runs/{problem_id}/artifact/papers/*.md
  - runs/{problem_id}/artifact/errors/*.md
  - runs/{problem_id}/artifact/citations.bib

- 不引入数据库（MVP 先不用 Redis/SQLite）。

## 8.2 在模块间如何流转

1. main.py 创建 Orchestrator 与 ProblemMemory。
2. Orchestrator 读取 ProblemMemory 的摘要索引，拼入 Agent prompt。
3. Agent 在当前阶段内多轮 tool call，短期记忆仅保留在 self.messages。
4. Agent 返回结构化文本后，Orchestrator 解析并写入 history.jsonl。
5. 若 Verifier 给出 verified_lemmas，Orchestrator 调 ProblemMemory.add_lemma 落盘。
6. state.json 在每轮末尾 save_state。
7. Finalizer 把 cite:路径转 References，并可导出 BibTeX。

## 8.3 数据格式示例

### state.json 示例

    {
      "problem_id": "PB-Advanced-001_20260331_120000_000001",
      "iteration_count": 2,
      "status": "RUNNING",
      "current_proof_path": "runs/PB-Advanced-001_20260331_120000_000001/",
      "last_verifier_decision": "MINOR_FLAW",
      "updated_at": "2026-03-31T12:34:56Z"
    }

### history.jsonl 单行示例

    {"timestamp":"2026-03-31T12:35:10Z","agent_node":"VERIFIER","turn_id":2,"decision":"MINOR_FLAW","verification_report_path":"runs/PB-Advanced-001_20260331_120000_000001/artifact/errors/002.md","verified_lemmas":["runs/PB-Advanced-001_20260331_120000_000001/artifact/lemmas/001.md"]}

### lemma markdown（三层）示例

    ---
    conditions:
      - n is positive integer
    conclusion: gcd(n, n+1)=1
    path: runs/PB-Advanced-001_20260331_120000_000001/artifact/lemmas/001.md
    ---

    ## Layer2-Proof
    Step 1. ...
    Step 2. ...

    ## Layer3-Source
    Source: self_proved
    Origin: Generator turn 1

### paper markdown（三层）示例

    ---
    id: arxiv-2501-12345
    theorem_summary: Under assumptions A,B, theorem T gives bound O(n log n)
    usage_condition:
      - assumption A
      - assumption B
    path: runs/PB-Advanced-001_20260331_120000_000001/artifact/papers/arXiv_2501_12345.md
    ---

    ## Layer2-ExtractedContent
    Theorem statement...
    Proof sketch...

    ## Layer3-ReferenceMeta
    arXiv_id: 2501.12345
    title: Sample Title
    authors: Alice; Bob
    url: https://arxiv.org/abs/2501.12345

---

## 9. 配置与可复现性

## 9.1 配置文件建议

建议在 settings.yaml 增加以下结构（示意）：

    provider: deepseek

    llm_defaults:
      thinking: true
      max_tokens: 16384
      connect_timeout_seconds: 30
      read_timeout_seconds: 600
      stream_max_retries: 2

    agent:
      max_turns: 3
      generator_max_tool_rounds: 5
      reviser_max_tool_rounds: 3
      verifier_max_tool_rounds: 8
      searcher_max_tool_rounds: 5

    memory:
      root_dir: runs
      auto_save_state_each_turn: true
      inject_summary_limit: 20

    reproducibility:
      random_seed: 42
      run_name_template: "{problem_id}_{timestamp}"
      freeze_prompt_version: true
      save_effective_config: true

---

## 10. 依赖项说明（必需库）

pydantic：定义 ProofState/ProblemMemory 元数据模型，避免字典字段漂移（场景见 src/core/state.py）。
tenacity：统一网络与模型调用重试，替代分散的手写重试逻辑（场景建议用于 src/models/llm_client.py）。
pyyaml：加载 settings.yaml 与 prompts.yaml（场景见 src/core/config.py）。
openai：统一调用 OpenAI 兼容接口（场景见 src/models/llm_client.py）。
httpx：配置细粒度超时、连接错误分类（场景见 src/models/llm_client.py）。
python-dotenv：从 .env 注入密钥（场景见 main.py）。
sympy：让 run_python 对符号推导更可靠（场景见 src/tools/code_executor.py）。
numpy：数值验证与快速实验（场景见 src/tools/code_executor.py）。
scipy：必要时补充科学计算函数（场景见 src/tools/code_executor.py）。

可选但强烈建议：

jsonschema：校验 Agent 标签输出结构，减少 parser 崩溃。
orjson：高频 JSONL 写入更快。

---

## 11. 你可能不熟悉的高级 Python 特性/库（每项一行）

dataclasses：用轻量对象承载 LLMResponse，减少样板代码并提高可读性（场景: src/models/llm_client.py）。
类型联合与可选类型：通过 str | None 让接口契约更明确（场景: src/core/orchestrator.py）。
Protocol 思维（建议引入）：用结构化接口约束 Pipeline/Logger/Finalizer，便于替换实现（场景: src/core/agent.py 适配器）。
上下文管理与 finally：确保临时资源一定释放（场景: src/tools/code_executor.py 删除临时文件）。
线程+join超时：把 LLM 摘要调用做硬超时隔离，避免卡死主流程（场景: src/utils/worklog_builder.py）。
Pydantic BaseModel：对日志与状态字段做运行时校验（场景: src/core/state.py）。
OpenAI function calling：把工具调用变成可结构化路由（场景: src/tools/registry.py 与 src/models/llm_client.py）。
YAML 环境变量替换：配置里直接用 ENV 占位符，部署更灵活（场景: src/core/config.py）。

---

## 12. 可直接落地的代码改造清单（按优先级）

## P0（先做，保证系统跑起来）

1. 新增 states/problem_memory.py 与 states/state.py 整理
   - 定义 ProblemMemory 类，并通过 `contextvars` 将其设置为当前题目的线程级全局单例/上下文变量。
   - 实现 init_dirs、save_state、load_state、append_history、add_lemma、add_paper、add_error。

2. 重构 src/core/orchestrator.py
   - 初始化 ProblemMemory 并设置到 ContextVar 中。
   - 全面废弃旧的 `pipeline.py`，改为在 Orchestrator 中直接实例化并调度 `agents/` 下的对象。
   - 每轮写 history.jsonl 与 state.json。
   - 解析 verifier 的 verified_lemmas 并调用 add_lemma。

3. 扩展 src/utils/parsing/parser.py
   - 新增 parse_lemmas_from_solution。
   - 新增 parse_verified_lemmas。
   - 新增 parse_citations。

4. 扩展 config/prompts/ 下的配置
   - Generator 强制 lemma 位置和 cite:文件路径规则。
   - Verifier 增加 verified_lemmas 输出契约。

## P1（增强可用性）

5. 新增 src/tools/artifact_reader.py
   - read_layer1(path)、read_layer2(path)、read_layer3(path)。
   - 参数只接收文件路径。

6. 新增 src/agents/searcher.py 与 src/tools/searcher_bridge.py
   - 支持 Generator 工具化调用 Searcher。
   - 去重逻辑优先查 ProblemMemory 的 papers 索引。

7. 增强 src/core/finalizer.py
   - 将 cite:路径 替换为编号引用 [1]。
   - 自动生成 References 与 citations.bib。

## P2（工程质量）

8. 新增 tests/test_problem_memory.py、tests/test_parser_contracts.py、tests/test_orchestrator_routing.py。
9. 在 run_imobench.py 增加 lemma_accept_rate、citation_coverage 两个指标。

---

## 13. 关键接口草案（可直接照着实现）

### 13.1 ProblemMemory

    class ProblemMemory:
        def __init__(self, problem_id: str, root_dir: str = "runs") -> None: ...
        def init_dirs(self) -> None: ...
        def save_state(self, state: dict) -> None: ...
        def load_state(self) -> dict: ...
        def append_history(self, event: dict) -> None: ...
        def add_lemma(self, lemma_markdown: str) -> str: ...
        def add_paper(self, paper_markdown: str, arxiv_id: str) -> str: ...
        def add_error(self, error_markdown: str) -> str: ...
        def list_layer1_summaries(self, kind: str) -> list[dict]: ...

### 13.2 Agent 基类

    class BaseAgent:
        def __init__(self, llm_client, tools: list[dict], max_tool_rounds: int) -> None: ...
        def reset_stage_memory(self) -> None: ...
        def run(self, payload: dict) -> dict: ...

### 13.3 Verifier 输出契约（建议）

    <verdict>CORRECT|MINOR_FLAW|CRITICAL_FLAW</verdict>
    <verification>...</verification>
    <verified_lemmas>
      <lemma>
        <name>Lemma 1</name>
        <proof>完整证明文本</proof>
        <source>self_proved|artifact_path</source>
      </lemma>
    </verified_lemmas>

---

## 14. 风险与防呆

- 风险1：LLM 不按标签输出或输出多个同名标签。
  - 防呆：parser 彻底舍弃最初的 `text.find`，改用纯正则 `re.finditer` 支持多重复数 `<lemma>` 解析，防丢数据；加两轮格式提醒重试。

- 风险2：工具调用过多导致延迟和费用暴涨。
  - 防呆：每个 Agent 设置 max_tool_rounds，超限强制收敛。Searcher 执行完毕后**仅返回**“成功下载，路径为 xxx”，强迫 Generator 在自己的下一轮思维中按需显式调用 `read_artifact`，精确控制 Token 开销。

- 风险3：引理误收录。
  - 防呆：Verifier 必须输出完整 proof，缺失则拒绝入库。

- 风险4：引用路径失效与幻觉。
  - 防呆：作为 Verifier 的一项隐式验证任务。在 Verifier 的 Prompt 中强制要求校验 `[cite:xxx]` 路径的有效性。如果发现造假或找不到文件，将其作为 FLAW 加入错误报告打回，交给 Reviser 或 Generator 修补。最后 Finalizer 仅负责转义而不再抛异常。

---

## 15. MVP 完成标准（Definition of Done）

满足以下条件即算 MVP 完成：

1. 可以完整跑通一题：Generator -> Verifier -> Reviser/Generator -> Final。
2. runs/{problem_id} 目录完整生成 state.json、history.jsonl、artifact 子目录。
3. Verifier 能输出 verified_lemmas，且 Orchestrator 能落盘成功引理。
4. Generator 能在引用时输出 cite:路径，Finalizer 能生成 References。
5. 至少 3 个核心单测通过（ProblemMemory、Parser、Routing）。
6. run_imobench.py 仍可运行，不破坏现有评测流程。

---

## 16. 与当前代码的对应关系（你可以从这里开始动手）

- 现有调度主干在 src/core/orchestrator.py，先从这里接入 ProblemMemory。
- 现有状态模型在 src/core/state.py，先把 history 从内存重度依赖改成文件主存。
- 现有标签解析在 src/utils/parser.py，先扩 lemma 与 citation 的解析函数。
- 现有工具入口在 src/tools/registry.py，先引入 artifact_reader 与 searcher_bridge。
- 现有最终输出在 src/core/finalizer.py，最后接引用展开。

这条路径对初学者最友好：
先改状态与解析，再改调度，再加新工具，最后再补 Searcher。

---

## 17. 结论

本方案把复杂需求拆成了一个可执行的 MVP 重构路线：

- 架构上，完成从函数链到 Agent 协作的升级。
- 工程上，完成每题隔离、可追踪、可复现的记忆体系。
- 研究上，完成引理沉淀、文献分层暴露与引用追踪。

你可以按 P0 -> P1 -> P2 顺序推进；每一步都能运行、都能验收，不会陷入“大改到一半跑不动”的状态。
