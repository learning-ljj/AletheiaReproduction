# AletheiaReproduction 架构重构设计（MVP v1）

## 0. 适用范围与目标

这份文档是给“当前仓库可直接落地”的重构方案，不追求一步到位做成完整多智能体平台，而是先做一个稳定、可运行、可扩展、可复现的最小可行版本（MVP）。

本方案完全基于当前代码现状梳理，核心代码参考：
- [src/core/orchestrator.py](src/core/orchestrator.py#L12)
- [src/core/agent.py](src/core/agent.py)
- [src/agents/verifier.py](src/agents/verifier.py)
- [src/core/state.py](src/core/state.py#L42)
- [src/models/llm_client.py](src/models/llm_client.py#L16)
- [src/tools/registry.py](src/tools/registry.py)

---

## 1. 生成正式文档前的前置分析（先讲大白话）

### 1.1 先说结论：你现在系统的真实形态

你现在的系统可以理解成：
1. 一个固定流程的总控器（Orchestrator）。
2. 三个“对象化 Agent 节点”按路由推进（Generator -> Verifier -> Reviser/Generator）。
3. 三个 Agent 都有工具能力，并按 stage allowlist 做权限隔离。
4. 状态已按题目落盘到 runs/{problem_id}（state.json、history.jsonl、artifact/*），支持问题级记忆与分层读取。

所以它目前是“固定主循环 + 多Agent 协作 + 问题级记忆”的 MVP 形态。

你要变成的是：

- 三个有状态的 Agent（Generator/Reviser/Verifier），每个阶段有自己的短期记忆。
- 每个问题有一个 ProblemMemory，负责公共记忆与落盘。
- 所有 Agent 都能调用不同的工具，Generator 还可以拉起 Searcher 子 Agent。
- 引理和论文都分层存储，按需读取，不把长文本一次性塞给模型。

一句话总结：
从“函数流水线”升级为“多Agent 协作 + 问题级记忆 + 分层知识加载”的可扩展框架。

### 1.2 实现你这个需求，最核心的模块有哪些

1. Orchestrator（总调度）
- 作用：只做流程推进，不做数学推理。
- 要做的事：创建 ProblemMemory、调用各 Agent、解析 verifier 输出、执行路由、落盘状态。

2. ProblemMemory（每题记忆中枢）
- 作用：每道题一个实例，统一管理 state.json、history.jsonl、artifact 文件（lemmas、papers、errors、citations），提供artifact 文件分层读取接口（只读第一层摘要、第二层正文、第三层来源）。
- 要做的事：保存 ProofState、保存引理、保存论文摘要与正文、保存错误报告。

3. GeneratorAgent（提出解题方案与引理）
- 作用：先产出结构化 solution，必须把候选引理写在 solution 前面，且用 lemma 标签块标出来。如果引用artifact中的知识，必须写 cite:文件路径。
- 要做的事：允许多轮工具调用；强制输出 cite 路径。

4. VerifierAgent（严审与分流）
- 作用：验证整份解答和其中 lemma，不负责补证明，引理没有完整证明就拒绝通过。把验证通过的引理放进 verified_lemmas 标签，交给 Orchestrator 入库。给出路由级别的结论（CORRECT/MINOR_FLAW/CRITICAL_FLAW）。
- 作用（新增）：当解答存在 cite 引用时，按需调用 CitationReviewer 子 Agent 做引用审查（路径存在性 + 引用内容正确性 + 元信息一致性）。
- 要做的事：输出 verdict、verification、verified_lemmas、citation_review；把可入库引理完整拷贝出来。

5. ReviserAgent（定点修补）
- 作用：按 verifier 报告修补，不要整篇重写。
- 要做的事：保留正确段落，只修有问题位置。

6. Searcher 子 Agent（文献助手）
- 作用：被 Generator 按需调用，做文献检索、清洗、提取、去重。
- 要做的事：把论文按三层结构写入 artifact/papers。

7. CitationReviewer 子 Agent（引用审查助手）
- 作用：被 Verifier 按需调用，逐条审查 [cite:路径] 的真实性与对应关系。
- 要做的事：输出逐条审查结果（通过/失败、证据、置信度、建议动作），回传给 Verifier 汇总成 citation_review。

8. Parser + Finalizer（协议解析与最终出稿）
- 作用：解析标签、提取 cite、把 cite 路径替换成标准引用编号，构造最终输出文件。
- 要做的事：构造 References 和可导出的 BibTeX，输出最终文档。

### 1.3 各模块核心工作流程（MVP）

1. main.py 接到题目，生成 run_id。
2. Orchestrator 初始化 ProblemMemory（按 problem_id 创建目录）。
3. Orchestrator 读取 ProblemMemory 的摘要（层1）拼进 Generator 输入。
4. Generator 进入工具循环；需要外部知识时调用 Searcher 子 Agent，执行“查询扩展 -> 多源检索 -> 去重排序 -> 逐篇 PDF 抽取 -> 分段摘要 -> 候选 claim 萃取”，并把论文三层内容写入 artifact/papers。
5. Generator 输出 solution（solution 开头含多个 lemma，正文中有 cite 路径）。
6. Reviser 按 verifier 报告修补，不进行整篇重写。
7. Verifier 验证整体与 lemma；若检测到 cite，则在 Verifier 内部调用 CitationReviewer 子 Agent 做引用审查。
8. Verifier 输出 verdict + verification + verified_lemmas + citation_review。
9. Orchestrator 解析 verified_lemmas，调用 ProblemMemory.add_lemma() 落盘。
10. 路由：
- CORRECT -> FINAL
- MINOR_FLAW -> REVISER
- CRITICAL_FLAW -> GENERATOR
11. FINAL 阶段把 artifact 文件中的 cite 路径替换为 [1][2]，自动生成 References，并附加 Verifier 给出的 citation_review 摘要。

### 1.4 显式需求与隐式需求

显式需求（你在 idea.md 中已经明确）：
1. ProofState 升级为 ProblemMemory 体系，按题隔离。
2. Generator/Reviser/Verifier 重构为有状态 Agent。
3. Generator 必须输出 lemma 标签；Verifier 必须输出 verified_lemmas。
4. 三层分层暴露（frontmatter 摘要、正文、来源元信息）。
5. Searcher 为独立子 Agent，负责去重与论文落盘。
6. 引用必须写 [cite:路径]，最终输出 References。

隐式需求（不做会反复踩坑）：
1. 输出协议必须统一，否则 parser 会频繁失败。
2. 文件写入需要原子性和路径安全检查，否则易出现损坏和越界读取。
3. 每个 Agent 需要 max_tool_rounds，防止工具死循环。
4. 状态要可恢复（state.json + history.jsonl）。
5. 提示词必须版本化，否则同代码不同结果不可对比。

### 1.5 当前代码中的高优先级风险（建议先修）

1. 日志路径存在兼容双写风险
- 当前主路径已迁移到 runs/{problem_id}/history.jsonl。
- 仍保留 legacy 镜像能力时，容易造成“看错日志来源”的运维混淆。
- 建议：默认关闭 legacy 镜像，仅在回归对比时显式开启。

2. parser 入口已收敛
- 当前统一入口为 [src/utils/parsing/parser.py](src/utils/parsing/parser.py)。
- 旧入口已删除，避免导入分裂和重复实现。
- 建议：新增解析能力时仅在 parsing 子包扩展。

### 1.6 三项目参考点是如何实现的（代码级拆解 + 迁移结论）

这部分对应你在 三项目参考点.md 中列出的“可借鉴点”，并明确“当前架构怎么接进去”。

#### 1.6.1 EvoScientist 的可借鉴点与实现方式

1. 主 Agent 调工具/技能/子代理的决策流程
- 实现位置：参考evoscientist/portable_agent_core/agent_framework.py。
- 关键机制：AgentRuntime.run() 以 strategy.next_action() 驱动循环，动作分为调用工具、委派子代理、结束；每步都会发 RunEvent 到 event sink。
- 如何搬到本项目：
  1. 在 src/agents/base.py 定义统一动作对象（Action）和回合事件对象（RunEvent）。
  2. 在 src/core/orchestrator.py 中只保留“阶段路由”，把“阶段内部动作决策”放回各 Agent。
  3. 在 src/utils/logging/logger.py 统一落盘事件，形成可回放轨迹。
- 注意事项：Orchestrator 只做跨阶段控制，不要把 Agent 内部中间推理塞回全局状态。
- 易犯错点：
  1. 让 Orchestrator 同时管理“阶段路由 + 工具调用细节”，导致职责回退到单体大函数。
  2. RunEvent 直接写入大段原文，造成 history 膨胀和回放困难。

2. 容错机制和自我修正
- 实现位置：参考evoscientist/portable_agent_core/resilience.py。
- 关键机制：retry_async（有界重试）、guarded_tool_call（工具异常保护）、run_with_self_correction（验证失败后自修正再执行）。
- 如何搬到本项目：
  1. 在 src/models/llm_client.py 增加统一重试策略（网络错误、429、超时）。
  2. 在 src/tools/registry.py 为工具执行增加 guarded wrapper，返回结构化错误而非抛异常中断。
  3. 在 Verifier/Generator 的结构化输出解析失败时触发一次“格式修复重试”。
- 注意事项：有副作用的工具（写文件、执行代码）必须做幂等保护，避免重试造成重复写入。
- 易犯错点：
  1. 无差别重试所有异常，导致逻辑错误被掩盖。
  2. 自修正回路未设置上限，引发隐性死循环。

3. 低耦合状态协议
- 实现位置：参考evoscientist/portable_agent_core/shared_types.py。
- 关键机制：AgentTask/SubAgentReport 等 DTO，把“运行时行为”和“领域对象”分离。
- 如何搬到本项目：
  1. src/memory/state.py 只承载领域状态（problem/workflow/task）。
  2. src/agents/* 使用独立的运行时 DTO（Action、ToolResult、SubAgentReport）。
  3. Parser 只解析协议字段，不直接写业务状态。
- 注意事项：DTO 版本要显式管理（如 schema_version），否则后续兼容会失控。
- 易犯错点：把 ProblemMemory 直接当消息总线使用，导致状态模型被瞬时字段污染。

#### 1.6.2 AutoResearchClaw 的可借鉴点与实现方式

1. 文献检索的完整链路
- 实现位置：参考AutoResearchClaw/stable_literature_search.py。
- 关键机制：expand_queries -> 多源请求 -> 重试与限流兜底 -> deduplicate_papers。
- 如何搬到本项目：
  1. 在 src/agents/searcher.py 固定执行全量链路：查询扩展、多源检索、去重重排、逐篇 PDF 抽取、候选 claim 萃取。
  2. 在 src/tools/search.py 暴露给 Generator 的工具桥接接口，输入 query_bundle，输出标准 paper 结构。
  3. 将 Layer1/Layer2/Layer3 一次性写入 artifact/papers，避免后续回填字段不一致。
- 注意事项：去重键顺序建议 DOI > arXiv ID > normalized title，且保留原始来源列表便于追溯。
- 易犯错点：
  1. 只按标题去重，误合并同名不同论文。
  2. 查询扩展过度导致召回噪声激增，反而稀释关键信息。

2. 引文真实性核验
- 实现位置：参考AutoResearchClaw/citation_review.py。
- 关键机制：verify_one_entry 采用 DOI -> OpenAlex -> arXiv -> title fallback 的级联验证；verify_citations 汇总核验结果并标注疑似幻引。
- 如何搬到本项目：
  1. 将核验能力做成 CitationReviewer 子 Agent（src/agents/citation_reviewer.py）。
  2. 在 src/agents/verifier.py 中仅在检测到 cite 时按需调用该子 Agent（类似 Generator 调 Searcher）。
  3. Verifier 汇总输出 citation_review（通过率、失败项、证据链、建议路由），Orchestrator 仅消费结果不直接执行核验。
- 注意事项：要同时检查“路径存在”与“引用断言是否被来源内容支撑”，不能只做文件存在性检查。
- 易犯错点：
  1. 只校验 cite 路径存在，不校验段落结论与来源内容的一致性。
  2. 把引用核验写进 Orchestrator，导致调度层职责膨胀。

3. 写作产物集中打包
- 实现位置：参考AutoResearchClaw/latex_pipeline.py 与 deliverables 约定。
- 关键机制：Markdown -> LaTeX -> 编译 -> 错误解析；最终集中导出论文、BibTeX、验证报告和清洗报告。
- 如何搬到本项目：
  1. MVP 先不强制 LaTeX 编译，但在 Finalizer 增加 manifest.json 汇总（solution、references、citation_review、errors）。
  2. 将引用与验证产物统一放在 runs/{problem_id}/artifact 下，路径固定。
- 注意事项：manifest 需要可机器读取，字段命名不要依赖自然语言。
- 易犯错点：把“展示文本”和“程序消费字段”混在一个文件，导致后续自动评测难以解析。

#### 1.6.3 ResearchClaw 的可借鉴点与实现方式

1. 状态对象化
- 实现位置：参考ResearchClaw/research_workflow_engine.py。
- 关键机制：ResearchProject/ResearchWorkflow/WorkflowTask/ResearchClaim/ResearchEvidence 等 dataclass + JsonStateStore。
- 如何搬到本项目：
  1. 在 src/memory/state.py 定义最小 typed state（ProblemState、StageState、TaskState）。
  2. 在 src/memory/problem_memory.py 实现单文件快照 + 原子写入 + 回读校验。
  3. 在 Orchestrator 每轮写入“状态快照 + 事件增量”。
- 注意事项：MVP 只引入最小必需字段，避免一开始引入过宽 schema。
- 易犯错点：
  1. 先设计过深对象图，导致开发早期大量字段长期为空。
  2. ID 规则不统一，后续 claim/evidence 无法可靠关联。

2. 状态随流程推进的管理思路
- 实现位置：参考ResearchClaw/research_workflow_engine.py 中 _recompute_workflow()、tick_workflow()、dashboard()。
- 关键机制：根据 task 状态重算 stage 状态，支持 blocked/running/completed。
- 如何搬到本项目：
  1. 保留当前主循环，新增 lightweight stage 重算函数（不引入完整图引擎）。
  2. 每轮依据 Verifier verdict 更新 stage 状态并写入 state.json。
  3. 提供最小 dashboard 视图用于调试（当前阶段、阻塞原因、最近错误）。
- 注意事项：状态重算必须“单一事实来源”，避免 stage 与 task 双向覆盖。
- 易犯错点：手动修改 stage 状态而不重算 task，造成状态漂移。

3. 文献到 claim/evidence 的结构化落盘
- 实现位置：record_literature_search()、record_paper_summary()。
- 关键机制：文献短名单、论文摘要、claim/evidence 关联对象一并写入状态。
- 如何搬到本项目：
  1. MVP 阶段将 claim/evidence 作为可选视图（claim_graph.json），不阻塞主流程。
  2. 仅对“Verifier 通过或部分通过”的条目生成 claim/evidence，降低噪声。
- 注意事项：claim 必须带来源路径与段落证据，避免不可追溯断言。
- 易犯错点：把未经验证的生成内容直接写入 claim 图，导致图谱污染。

---

## 2. 你可能不熟悉的高级 Python 特性/第三方库（一行版）

按照你要求的格式：库/语言特性：一句话说明在本项目中的作用（并给出典型使用场景或代码片段链接）。

1. dataclasses：用轻量结构体承载模型返回对象，减少样板代码并提升可读性（场景：[src/models/llm_client.py](src/models/llm_client.py#L16)）。
2. pydantic：给状态模型做运行时强校验，防止字段错类型导致脏数据（场景：[src/core/state.py](src/core/state.py#L42)）。
3. typing 联合类型（如 str | None）：把接口约束写清楚，降低函数调用误用（场景：[src/core/orchestrator.py](src/core/orchestrator.py)）。
4. OpenAI Function Calling：让工具调用变成结构化函数路由，便于审计和扩展（场景：[src/models/llm_client.py](src/models/llm_client.py#L224)）。
5. subprocess：把 run_python 放进子进程隔离执行，降低主进程被污染风险（场景：[src/tools/code_executor.py](src/tools/code_executor.py#L13)）。
6. threading + join 超时：给离线摘要调用加硬超时，避免工作日志构建卡死（场景：[src/utils/logging/worklog_builder.py](src/utils/logging/worklog_builder.py#L116)）。
7. pathlib：统一路径拼接和跨平台目录管理，减少 Windows/Linux 差异问题（场景：[src/utils/logging/logger.py](src/utils/logging/logger.py#L11)）。
8. yaml 配置替换环境变量：把密钥和模型参数从代码中剥离，便于复现实验（场景：[src/core/config.py](src/core/config.py#L8)）。
9. tenacity（建议新增）：把 LLM/网络重试策略标准化，替代分散手写重试逻辑（建议场景：重构 [src/models/llm_client.py](src/models/llm_client.py)）。
10. contextvars（建议新增）：实现“同题共享、跨题隔离”的 ProblemMemory 上下文传递（建议场景：新增 src/memory/problem_memory.py）。

---

## 3. MVP 架构设计文档（正式版）

### 3.1 设计目标与非目标

目标：
1. 保持现有主流程可运行。
2. 在最小改动下引入 ProblemMemory 与 Agent 对象化。
3. 支持引理入库、文献分层、引用追踪。
4. 保证可复现（状态、日志、配置、模型版本都可追踪）。

非目标（MVP 先不做）：
1. 跨题共享知识库。
2. 多进程并发调度器。
3. 完整数据库（Redis/SQLite）持久化。

### 3.2 核心模块划分（按功能域）

1. 数据加载
- 负责读取题目与 benchmark。
- 现有文件：[src/utils/evaluation/data_loader.py](src/utils/evaluation/data_loader.py)

2. 预处理
- 负责 prompt 注入、上下文摘要拼接、输入标准化。
- 建议新增：src/core/context_builder.py

3. 核心逻辑处理
- 负责调度、路由、状态推进。
- 现有文件：[src/core/orchestrator.py](src/core/orchestrator.py)

4. 模型接口
- 负责 chat、tool-calling、流式输出、重试。
- 现有文件：[src/models/llm_client.py](src/models/llm_client.py)

5. 推理管道
- 负责 Generator/Reviser/Verifier/Searcher 的 run。
- 当前已对象式：[src/agents/base.py](src/agents/base.py)、[src/agents/generator.py](src/agents/generator.py)、[src/agents/reviser.py](src/agents/reviser.py)、[src/agents/verifier.py](src/agents/verifier.py)、[src/agents/searcher.py](src/agents/searcher.py)
- 由 [src/core/agent.py](src/core/agent.py) 统一装配运行时依赖。

6. 结果解析
- 负责 XML 标签解析、lemma 解析、cite 解析。
- 现有文件：[src/utils/parsing/parser.py](src/utils/parsing/parser.py)

7. 自动化评估
- 负责 answerbench/proofbench/gradingbench 批评测。
- 现有文件：[scripts/run_imobench.py](scripts/run_imobench.py)

8. 日志与监控
- 负责 raw 事件记录与 worklog 构建。
- history.jsonl 原始事件。
- state.json 快照。
- artifact 实体内容。
- 现有文件：[src/utils/logging/logger.py](src/utils/logging/logger.py)、[src/utils/logging/worklog_builder.py](src/utils/logging/worklog_builder.py)

9. 配置管理
- 负责 settings 与 prompts 加载。
- 现有文件：[src/core/config.py](src/core/config.py)、[config/settings.yaml](config/settings.yaml)、[config/prompts](config/prompts)

10.  测试
- 负责单测、集成测试、实时监控。
- 现有文件：[tests/realtime_e2e_monitor.py](tests/realtime_e2e_monitor.py)

### 3.3 整体目录结构树（当前精确版 + MVP 目标版）

#### 3.3.1 当前目录结构（精确到 py 与关键配置）

~~~text
AletheiaReproduction/
  main.py
  config/
    default.yaml
    settings.yaml
    prompts/
      generator.yaml
      reviser.yaml
      verifier.yaml
      searcher.yaml
      citation_reviewer.yaml
      final.yaml
  scripts/
    run_imobench.py
    run_proofbench_advanced.py
  src/
    __init__.py
    agents/
      __init__.py
      base.py
      generator.py
      reviser.py
      verifier.py
      searcher.py
      citation_reviewer.py
    core/
      __init__.py
      agent.py
      config.py
      context_builder.py
      finalizer.py
      orchestrator.py
      state.py
    memory/
      __init__.py
      state.py
      problem_memory.py
    models/
      __init__.py
      llm_client.py
    tools/
      __init__.py
      artifact_reader.py
      code_executor.py
      registry.py
      search.py
    utils/
      __init__.py
      parsing/
        __init__.py
        parser.py
        reference_builder.py
      evaluation/
        __init__.py
        data_loader.py
        evaluator.py
      logging/
        __init__.py
        logger.py
        raw_log_reader.py
        worklog_builder.py
  tests/
    realtime_e2e_monitor.py
    test_agent_tool_policy.py
    test_citation_review.py
    test_finalizer_reference.py
    test_infrastructure.py
    test_orchestrator_route.py
    test_parser_contract.py
    test_pipeline_contracts.py
    test_problem_memory.py
    test_searcher_dedup.py
    test_stage_integration.py
  bin/
    tool/
      _http_utils.py
      web_search.py
      wiki_search.py
~~~

#### 3.3.2 每个当前文件做什么（通俗说明）

1. [main.py](main.py)：CLI 入口，读取题目、创建 Agent、输出结果并可生成 worklog。
2. [scripts/run_imobench.py](scripts/run_imobench.py)：批量评测三类数据集并输出统计结果。
3. [scripts/run_proofbench_advanced.py](scripts/run_proofbench_advanced.py)：只跑 proofbench 的 advanced 子集。
4. [src/core/agent.py](src/core/agent.py)：当前门面类，装配 AgentRuntime / Orchestrator / Logger / Finalizer。
5. [src/core/orchestrator.py](src/core/orchestrator.py)：当前核心状态机，负责节点调用和路由。
6. [src/agents/base.py](src/agents/base.py)：对象化 Agent 基类与阶段执行循环（具体节点在 src/agents/*.py）。
7. [src/core/state.py](src/core/state.py)：当前状态模型（ProofState/VerificationLog/枚举）。
8. [src/core/config.py](src/core/config.py)：配置和 prompt 加载，支持环境变量替换。
9. [src/core/finalizer.py](src/core/finalizer.py)：构造最终输出文本（成功/失败/部分进展）。
10. [src/models/llm_client.py](src/models/llm_client.py)：LLM 客户端，支持 thinking、流式、工具调用。
11. [src/tools/registry.py](src/tools/registry.py)：工具 schema 注册与统一执行路由。
12. [src/tools/code_executor.py](src/tools/code_executor.py)：Python 子进程沙箱执行。
13. [src/utils/parsing/parser.py](src/utils/parsing/parser.py)：输出标签解析与 verdict 解析。
14. [src/utils/logging/logger.py](src/utils/logging/logger.py)：JSONL 事件追加与 artifact markdown 保存。
15. [src/utils/logging/raw_log_reader.py](src/utils/logging/raw_log_reader.py)：读取 raw JSONL。
16. [src/utils/logging/worklog_builder.py](src/utils/logging/worklog_builder.py)：把 JSONL 转成可读 markdown 报告。
17. [src/utils/evaluation/data_loader.py](src/utils/evaluation/data_loader.py)：读取 benchmark CSV 和回填 ground truth。
18. [src/utils/evaluation/evaluator.py](src/utils/evaluation/evaluator.py)：短答和证明完整度的简单评测。
19. [tests/realtime_e2e_monitor.py](tests/realtime_e2e_monitor.py)：E2E 实时监控脚本。
20. [bin/tool/_http_utils.py](bin/tool/_http_utils.py)：网络抓取重试与 SSL 降级工具。
21. [bin/tool/web_search.py](bin/tool/web_search.py)：arXiv 搜索和 LaTeX 抽取。
22. [bin/tool/wiki_search.py](bin/tool/wiki_search.py)：Wikipedia 检索与清洗。

#### 3.3.3 MVP 目标目录结构（建议落地版）

~~~text
AletheiaReproduction/
  main.py
  architecture_v1.md
  requirements.txt
  config/
    settings.yaml                       # 保留并扩展：provider、timeouts、agent与memory配置
    prompts/                            # 新增：存放各agent的prompt配置
      generator.yaml                    # 新增：Generator的prompt
      reviser.yaml                      # 新增：Reviser的prompt
      verifier.yaml                     # 新增：Verifier的prompt
      citation_reviewer.yaml            # 新增：CitationReviewer的prompt
      searcher.yaml                     # 新增：Searcher的prompt
    default.yaml                    # 新增：复现实验默认配置
  runs/                             # 新增：每题隔离产物根目录
    {problem_id}/
      solution.md
      history.jsonl
      state.json
      artifact/
        lemmas/
        papers/
        errors/
        citations.bib

  src/
    __init__.py

    core/
      __init__.py
      orchestrator.py               # 重构：调度、路由、状态管理
      config.py
      finalizer.py                  # 重构：引用替换 + References + 最终输出
      context_builder.py            # 新增：组装 agent 输入上下文

    agents/                         # 新增：对象化 Agent 层
      __init__.py
      base.py                     # 新增：通用 Agent 基类（messages、tool loop、reset）
      generator.py                # 新增：生成解答与 lemma
      reviser.py                  # 新增：修订解答与 lemma
      verifier.py                 # 新增：验证解答与 lemma
      citation_reviewer.py        # 新增：引用审查子 Agent（由 Verifier 按需调用）
      searcher.py                 # 新增：检索、论文清洗、内容提取与检索去重

    memory/                         # 新增：问题级记忆层
      __init__.py
      state.py                      # 轻量 ProofState
      problem_memory.py

    models/
      __init__.py
      llm_client.py

    tools/
      __init__.py
      code_executor.py
      registry.py                   # 重构：按 agent 角色分别分配工具
      code_executor.py                  # 保留
      artifact_reader.py                # 新增：按层读取 markdown 第二层（详细证明）
      search.py                # 新增：Generator 调用 Searcher 的工具桥接

    utils/
      __init__.py
      parsing/                          # 新增：解析器与处理相关
        __init__.py
        parser.py                     # 重构：多 lemma / verified_lemmas / cite 解析
        reference_builder.py          # 新增：cite -> [1] 与 BibTeX
        reference_reader.py                # 新增：按层读取 markdown 第三层（引用信息）
      evaluation/                       # 新增：测试数据加载与评估
        __init__.py
        data_loader.py
        evaluator.py
      logging/                          # 新增：日志与事件记录相关
        __init__.py
        logger.py                     # 重构：改由 ProblemMemory 管理路径
        raw_log_reader.py
        worklog_builder.py

  scripts/
    run_imobench.py
    run_proofbench_advanced.py

  tests/
    realtime_e2e_monitor.py
    test_problem_memory.py          # 新增
    test_parser_contract.py         # 新增
    test_orchestrator_route.py      # 新增
    test_citation_review.py         # 新增
    test_searcher_dedup.py          # 新增
~~~ 

### 3.4 状态与上下文管理（存储位置、流转、数据样例）

#### 3.4.1 存储在哪里

1. 内存态
- Agent 阶段短期记忆：self.messages（每阶段 reset）。
- 调度态：当前 turn 的路由信息。

2. 文件态
- runs/{problem_id}/state.json：当前题目快照状态。
- runs/{problem_id}/history.jsonl：原始事件流。
- runs/{problem_id}/artifact/lemmas/*.md：验证通过引理。
- runs/{problem_id}/artifact/papers/*.md：文献三层信息。
- runs/{problem_id}/artifact/errors/*.md：错误分析报告。
- runs/{problem_id}/artifact/citations.bib：最终导出引用。

- 不引入数据库（MVP 先不用 Redis/SQLite）。

#### 3.4.2 模块间如何流转

~~~text
main.py
  -> Orchestrator.run(problem_id, problem_text)
     -> ProblemMemory.init()
    -> Generator.run(context + layer1 summaries)
      -> (optional) Searcher.run(query_bundle)
        -> expand_queries
        -> multi_source_search
        -> deduplicate + rerank
        -> pdf_extract + section_parse + claim_candidate_extract
        -> ProblemMemory.add_paper(...)
    -> Verifier.run(problem + solution)
        -> parse cites from solution
        -> (if cites exist) CitationReviewer.run(cite_entries + cited_claim_spans)
        -> output: verdict + verification + verified_lemmas + citation_review
     -> Orchestrator.parse_verified_lemmas()
        -> ProblemMemory.add_lemma(...)
     -> route to Reviser or Generator
     -> Finalizer(reference_builder) 生成最终可读输出，把 cite:路径转 References，并可导出 BibTeX。
~~~

#### 3.4.3 关键数据格式示例

state.json 示例：

~~~json
{
  "problem_id": "PB-Advanced-001_20260331_153000",
  "iteration_count": 2,
  "status": "RUNNING",
  "current_solution_path": "runs/PB-Advanced-001_20260331_153000/solution.md",
  "last_verifier_decision": "MINOR_FLAW",
  "updated_at": "2026-03-31T15:31:08Z"
}
~~~

history.jsonl 单条事件示例：同现有的data\logs\下的生成逻辑，只是观察运行过程和结果，方便修改代码的暂时性实现。

~~~

lemma 文档（三层）示例：

~~~markdown
---
summary: 若 n 为正整数，则 gcd(n, n+1)=1
conditions:
  - n is positive integer
conclusion: gcd(n, n+1)=1
source: self_proved
---

## Layer2-Proof
Step 1. 设 d 同时整除 n 与 n+1，则 d 整除 (n+1)-n=1。
Step 2. 因此 d=1，故 gcd(n,n+1)=1。

## Layer3-Source
Source: generator
Reference: self_proved
~~~

paper 文档（三层）示例：

~~~markdown
---
arxiv_id: 2501.12345
summary: 在条件 A,B 下，得到界 O(n log n)
conditions:
  - assumption A
  - assumption B
conclusion: bound O(n log n)
---

## Layer2-Extracted
Theorem ...
Proof ...

## Layer3-Source
title: Sample Paper
authors: Alice; Bob
url: https://arxiv.org/abs/2501.12345
~~~

### 3.5 配置

在你选定的技术路线下（渐进混合编排 + 全量检索链路 + 软引用门控），建议把配置拆成“运行控制 + 检索核验 + 提示词版本”三组。

settings.yaml 建议新增字段：

~~~yaml
orchestrator:
  mode: hybrid_loop                  # hybrid_loop | full_workflow
  enable_stage_state: true
  max_turns: 8
  event_log: true

resilience:
  llm_retry_max_attempts: 3
  tool_retry_max_attempts: 2
  max_tool_rounds: 4
  tool_timeout_seconds: 30

retrieval:
  depth: full                        # lite | full
  enable_query_expansion: true
  providers: [semantic_scholar, arxiv]
  max_results_per_query: 20
  pdf_extract:
    enabled: true
    max_pages: 24
  dedup:
    keys: [doi, arxiv_id, normalized_title]

verifier:
  citation_review:
    enabled: true
    trigger_when_citation_present: true
    subagent: citation_reviewer
    mode: soft                        # soft | hard
    checks:
      - path_exists
      - claim_source_match
      - metadata_consistency
      - doi_openalex_arxiv_title_cascade
    warn_only: true
    hard_fail_threshold: 0.4          # 仅在 hard 模式启用
    max_tool_rounds: 3

repro:
  snapshot_prompts: true
  snapshot_settings: true
~~~

prompts 配置建议：
1. generator.yaml：显式约束“若使用外部知识，必须输出 [cite:path]”。
2. verifier.yaml：显式约束“verified_lemmas 必须给出完整可复用证明文本”，且在存在 cite 时必须调用 CitationReviewer。
3. citation_reviewer.yaml：显式约束“每条 cite 都要输出证据、结论、置信度、失败原因”。
4. searcher.yaml：要求输出结构化 paper 记录（metadata + Layer2 摘要 + Layer3 来源）。
5. reviser.yaml：只修故障点，不重写全稿。

### 3.6 依赖项说明（必需库 + 为什么需要）

1. openai：统一访问 OpenAI 兼容模型接口，是所有 Agent 的主推理入口。
2. pydantic：保证状态模型和结构化输出字段稳定，降低运行期脏数据风险。
3. pyyaml：加载 settings 与 prompts，是配置驱动架构基础。
4. httpx：网络超时、连接错误分类、重试协作依赖。
5. python-dotenv：从 .env 注入密钥与 provider 配置，便于本地开发。
6. sympy：在 run_python 中做符号推导与数学验证，避免纯字符串推理。
7. numpy：数值验证和快速试算。
8. scipy：部分高阶数值/科学计算场景支持。
9. tenacity（建议新增）：把 API 与网络重试策略标准化，减少重复手写重试代码。
10. pypdf（建议新增）：逐篇 PDF 正文抽取与分段摘要所需。

### 3.7 从顶向下的最小改造路线（建议顺序）

P0（必须先做，保证主流程稳定）
1. 调整文件架构。
2. 新增 src/memory/problem_memory.py和state.py，实现 state/history/artifact 统一落盘。
3. 扩展 parser 支持多 lemma、verified_lemmas、cite 提取。
4. 新增运行协议对象（Action/RunEvent），并把 Orchestrator 每轮关键事件写入 history。
5. Orchestrator 接入 ProblemMemory，并在每轮保存 state 和 history。

P1（增强可用性）
1. 新建 src/agents，逐步替换 pipeline 函数为对象化 Agent。
2. 引入 search.py + searcher_agent，实现全量文献链路（查询扩展、检索、去重、PDF 抽取、摘要、候选 claim）。
3. 新增 src/agents/citation_reviewer.py，并在 src/agents/verifier.py 中按需调用，实现引用审查子代理链路。
4. Finalizer 接入 reference_builder，输出标准 References 与 citation_review 摘要。

P2（增强可评测与可运维）
1. 新增关键单测（memory/parser/routing/dedup/citation_reviewer/verifier_citation_route）。
2. 批评测脚本切换到 runs 目录读取日志。
3. 追加 dashboard.json、claim_graph.json 视图，吸收 ResearchClaw 的可观测性能力。
4. 输出 run_meta.json 和 prompt 快照，完成复现闭环。

### 3.8 可直接落地的代码/文件建议（最小骨架）

#### 建议一：新增 src/memory/problem_memory.py

~~~python
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProblemMemory:
    problem_id: str
    root_dir: str = "runs"

    def __post_init__(self):
        self.base = Path(self.root_dir) / self.problem_id
        self.artifact = self.base / "artifact"
        self.lemmas = self.artifact / "lemmas"
        self.papers = self.artifact / "papers"
        self.errors = self.artifact / "errors"
        self.state_file = self.base / "state.json"
        self.history_file = self.base / "history.jsonl"

    def init_dirs(self) -> None:
        for p in [self.base, self.artifact, self.lemmas, self.papers, self.errors]:
            p.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self.state_file.write_text("{}\n", encoding="utf-8")

    def save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def append_history(self, event: dict) -> None:
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
~~~

#### 建议二：在 src/utils/parsing/parser.py 增加多标签提取

~~~python
import re

def extract_xml_tags(text: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(text or "")]

def parse_citations(text: str) -> list[str]:
    items = re.findall(r"\[cite:([^\]]+)\]", text or "")
    seen = set()
    out = []
    for x in items:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out
~~~

#### 建议三：Orchestrator 每轮统一写状态与事件

~~~python
# 伪代码示意
memory = ProblemMemory(problem_id)
memory.init_dirs()

for turn in range(max_turns):
    gen_text = generator.run(...)
    memory.append_history({...})

    ver_text = verifier.run(...)
    memory.append_history({...})

    verified = parse_verified_lemmas(ver_text)
    for lemma in verified:
        memory.add_lemma(lemma["markdown"])

    memory.save_state({...})
~~~

#### 建议四：Verifier 输出契约（XML 标签 + 结构化字段）

    <verdict>CORRECT|MINOR_FLAW|CRITICAL_FLAW</verdict>
    <verification>...</verification>
    <lemma>
      <introduction>引理简述</introduction>
      <theorem>引理内容（包含使用条件、结论）</theorem>
      <proof>完整证明文本</proof>
      <source>self_proved|artifact_path</source>
    </lemma>
    <lemma>
      <introduction>引理简述</introduction>
      <theorem>引理内容（包含使用条件、结论）</theorem>
      <proof>完整证明文本</proof>
      <source>self_proved|artifact_path</source>
    </lemma>

#### 建议五：新增 src/agents/citation_reviewer.py（由 Verifier 按需调用）

~~~python
class CitationReviewerAgent(BaseAgent):
  def review_one(self, cite_entry: dict, claim_span: str) -> dict:
    # 路径存在性、引用片段一致性、元信息一致性、外部级联核验
    # 返回 {passed, evidence, confidence, failure_reason}
    ...

  def review_all(self, cites: list[dict], claim_spans: list[str]) -> dict:
    # 返回 {summary, items, severity_suggestion}
    ...
~~~

#### 建议六：在 Verifier 中按需调用 CitationReviewer

~~~python
# 伪代码示意
cites = parse_citations(solution_text)
citation_review = None
if cites:
  citation_review = citation_reviewer.review_all(cites, claim_spans)

verifier_output = {
  "verdict": verdict,
  "verification": verification,
  "verified_lemmas": verified_lemmas,
  "citation_review": citation_review,
}
~~~

#### 建议七：在 Searcher 中固定全量链路输出契约

~~~python
{
  "query": "...",
  "papers": [
    {
      "paper_id": "...",
      "title": "...",
      "layer1_summary": "...",
      "layer2_extract": "...",
      "layer3_source": {"doi": "...", "arxiv_id": "...", "url": "..."},
      "candidate_claims": ["..."]
    }
  ]
}
~~~

### 3.9 MVP 验收标准（你可以照这个打勾）

1. 每道题都生成 runs/{problem_id}/state.json 和 history.jsonl。
2. Verifier 输出的 verified_lemmas 能被解析并落盘到 artifact/lemmas。
3. Generator 的 cite 路径在最终输出被替换成 [1] 样式，并有 References。
4. Searcher 默认执行全量链路（检索、去重、PDF 抽取、摘要、candidate_claims），且同一 arXiv ID 二次检索不重复写 paper 文件。
5. Verifier 在存在 cite 时会调用 CitationReviewer，并输出 citation_review；未通过项会进入 warning，但不阻塞最终出稿（soft mode）。
6. 在 max_turns 与 max_tool_rounds 内流程稳定收敛，不出现无限工具调用。

### 3.10 三项目可借鉴点到本架构的映射总表

1. EvoScientist -> 运行时编排能力
- 迁移：Action/RunEvent + resilience wrapper。
- 落位：src/core/orchestrator.py、src/models/llm_client.py、src/agents/base.py。

2. AutoResearchClaw -> 检索与引用可信能力
- 迁移：全量文献链路 + 级联引用核验（由 Verifier 调用 CitationReviewer，软门控）。
- 落位：src/agents/searcher.py、src/tools/search.py、src/agents/verifier.py、src/agents/citation_reviewer.py、src/utils/parsing/reference_builder.py。

3. ResearchClaw -> typed 状态与阶段可观测性
- 迁移：ProblemMemory typed state + stage 重算 + dashboard/claim_graph 视图。
- 落位：src/memory/state.py、src/memory/problem_memory.py、src/core/orchestrator.py。

---

## 4. 给你的执行建议（按周推进）

第 1 周：先做 P0（尤其是 ProblemMemory 与 parser 升级），保证“能跑且能存”。
第 2 周：做 P1（Agent 对象化 + Searcher bridge + 引用处理），实现你要的核心体验。
第 3 周：做 P2（测试与复现闭环），把实验可信度补齐。

如果你希望，我下一步可以直接基于这份文档，帮你生成第一批可运行代码补丁（先做 P0）。
