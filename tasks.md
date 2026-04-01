# AletheiaReproduction 执行任务单 v5（按反馈修订）

目标：
1. 先完成对齐审计，再执行可落地改造。
2. 任务顺序严格按依赖推进，不跳阶段。
3. 每个任务都给出改动文件、操作步骤、验收命令、完成标准。
4. 将三个参考项目的实现作为本项目代码实现参考，明确借鉴点、落位位置与迁移避坑。
5. 保证任务按合理顺序逐步实现。

---

## 0. 执行输入与强约束

1. 对齐输入文档：`architecture.md`、`idea.md`、`参考evoscientist/portable_agent_core/README.md`、`参考AutoResearchClaw/README.md`、`参考ResearchClaw/README.md`。
2. 替换策略：被新架构替代的旧能力不做回退/降级分支，直接替换。
3. 检索功能实现策略：停止推进“旧 websearch 工具归位”；统一建设 SearcherAgent 检索链。
4. 顺序策略：严格按本任务单顺序推进，未通过当前阶段验收，不进入下一阶段。
5. 可验证策略：每个任务必须给出输入输出、伪代码、验收命令、完成标准。
6. 架构对齐原则:每个任务都必须对应 architecture.md 和 idea.md 的明确需求点。
7. 协议先行原则:先固定输出协议与解析，再扩展Agent 和子代理能力。

---

## 1. 三参考项目借鉴映射（怎么借鉴、借鉴什么、避开什么坑）

### 1.1 EvoScientist 借鉴映射

1. 借鉴内容：主 Agent 的动作循环（工具调用/子代理委派/结束）、受保护工具调用、重试与自修正上限。
2. 本项目落位：`src/agents/base.py`、`src/tools/registry.py`、`src/core/orchestrator.py`。
3. 借鉴方式：
   - 在 BaseAgent 中实现阶段内 `messages` 与 `max_tool_rounds`。
   - 在工具执行路径加入 guarded wrapper，错误转结构化结果而非抛异常中断。
   - 对解析失败仅允许有界修复重试，避免无限循环。
4. 避坑要点：
   - 不要让 Orchestrator 管理工具细节，Orchestrator 只做阶段路由。
   - 不要无限重试，必须设置严格上限。
   - 不要把大段推理原文写入事件流，避免 history 膨胀。

### 1.2 AutoResearchClaw 借鉴映射

1. 借鉴内容：稳定检索链路（查询扩展-多源检索-去重-抽取-落盘）、引用核验级联、最终产物打包思路。
2. 本项目落位：`src/agents/searcher.py`、`src/tools/search.py`、`src/agents/citation_reviewer.py`、`src/utils/reference_builder.py`、`src/core/finalizer.py`。
3. 借鉴方式：
   - SearcherAgent 固定全链路输出，不允许“只检索不落盘”。
   - CitationReviewer 在 Verifier 检测到 cite 时按需触发并返回结构化审查结果。
   - Finalizer 输出 references、bibtex 与 manifest 摘要。
4. 避坑要点：
   - 去重不能只按标题，顺序采用 DOI > arXiv ID > normalized title。
   - 不能只校验路径存在，必须核验“引用断言与来源内容一致性”。
   - 不要用模板降级掩盖失败，失败需结构化暴露。

### 1.3 ResearchClaw 借鉴映射

1. 借鉴内容：typed state、JSON 持久化、阶段状态重算、可观测视图。
2. 本项目落位：`src/memory/state.py`、`src/memory/problem_memory.py`、`src/core/orchestrator.py`。
3. 借鉴方式：
   - 将状态快照与事件增量分离存储（state.json + history.jsonl）。
   - 每轮依据 verdict 重算阶段状态并持久化。
   - 输出最小可观测字段，支持定位阻塞原因。
4. 避坑要点：
   - 初期 schema 不要过宽，先最小可用字段。
   - ID 规则必须统一，否则 lemma/citation 关联失真。
   - 不要混淆“运行时临时字段”和“长期状态字段”。

---

## 2. 任务总览（合理顺序）

1. Phase A：替换旧检索入口 + 状态底座（A10-A14）
2. Phase B：协议与解析（B20-B23）
3. Phase C：Agent 化与调度主链替换（C30-C34）
4. Phase D：Searcher 与 CitationReviewer 子代理（D40-D43）
5. Phase E：Finalizer 与引用导出（E50-E53）
6. Phase F：测试、评测、E2E 与最终门禁（F60-F64）

---

## 3. 详细任务单（可直接执行）

### Phase A：替换旧检索入口 + 状态底座

#### A10 替换旧检索入口（不归位 web_search）

1. 对齐依据：`architecture.md` 1.5（高优先级风险）、`idea.md` 工具层问题描述。
2. 借鉴来源：AutoResearchClaw（稳定检索链替代旧工具拼接）。
3. 目的/作用：移除 `src.tools.registry` 对旧 `web_search/wiki_search` 的运行时依赖，改为 Searcher 工具桥接入口。
4. 输入输出：
   - 输入：`function_name: str`, `arguments: dict`
   - 输出：`str`（工具结果或结构化错误）
5. 内部伪代码：
   - step1: 删除 registry 中对旧检索模块 import。
   - step2: 增加 `call_searcher` 占位工具 schema 与执行分发。
   - step3: 保留数学执行工具与通用错误封装。
6. 修改文件：
   - `src/tools/registry.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -c "import src.tools.registry as r; print('ok')"`
8. 完成标准：导入成功且不再依赖 `src.tools.web_search`、`src.tools.wiki_search`。

#### A11 新建 typed 状态模型

1. 对齐依据：`architecture.md` 1.4、1.6.3；`idea.md` 1.1。
2. 借鉴来源：ResearchClaw（typed state + 持久化）。
3. 目的/作用：将状态快照从历史流水中抽离，建立可校验状态模型。
4. 输入输出：
   - 输入：状态字典
   - 输出：`ProblemSnapshot`、`StageSnapshot` 对象与字典互转
5. 内部伪代码：
   - step1: 定义最小字段（problem_id, iteration_count, status, last_decision）。
   - step2: 实现 `to_dict/from_dict`。
   - step3: 非法字段抛结构化校验异常。
6. 修改文件：
   - 新建 `src/memory/state.py`
   - `src/core/state.py`（桥接或兼容导出）
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_problem_memory.py -k snapshot -q`
8. 完成标准：状态模型序列化和反序列化通过，类型错误可诊断。

#### A12 ProblemMemory 全量实现

1. 对齐依据：`architecture.md` 1.2、1.3；`idea.md` 1.1、1.2。
2. 借鉴来源：ResearchClaw（JsonStateStore 思路）。
3. 目的/作用：建立每题独立存储中枢，统一管理 state/history/artifact。
4. 输入输出：
   - 输入：`problem_id`, `state`, `event`, `lemma/paper/error/bib` 内容
   - 输出：`runs/{problem_id}` 下完整文件结构
5. 内部伪代码：
   - step1: `init_dirs` 创建 state/history/artifact 子目录。
   - step2: `save_state/load_state/merge_state` 原子读写。
   - step3: `append_event/read_events` 维护事件流。
   - step4: `add_lemma/add_paper/add_error/save_bibtex` 统一落盘。
6. 修改文件：
   - 新建 `src/memory/problem_memory.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_problem_memory.py -q`
8. 完成标准：每题目录完整、重复写入幂等、无脏写。

#### A13 Orchestrator 接管状态与事件

1. 对齐依据：`architecture.md` 1.2、1.3；`idea.md` 1.1。
2. 借鉴来源：ResearchClaw（流程推进与状态重算）。
3. 目的/作用：Orchestrator 成为唯一状态推进与事件写入入口。
4. 输入输出：
   - 输入：problem_text 与各 Agent 输出
   - 输出：每轮 state 快照与 event 增量
5. 内部伪代码：
   - step1: run 开始初始化 ProblemMemory。
   - step2: 每阶段结束写入 event。
   - step3: 每轮结束写入 state。
6. 修改文件：
   - `src/core/orchestrator.py`
   - `src/core/agent.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_orchestrator_route.py -k persist -q`
8. 完成标准：success/partial/fail 三类结束路径均有 state/history。

#### A14 日志链路切换到 runs（不保留 data/logs 回退）

1. 对齐依据：`architecture.md` 1.5；`idea.md` 1.1。
2. 借鉴来源：ResearchClaw（状态与日志统一持久化）。
3. 目的/作用：将日志读取与构建链路全部迁移至 `runs/{problem_id}`。
4. 输入输出：
   - 输入：`problem_id`, `event`
   - 输出：`runs/{problem_id}/history.jsonl` 与 worklog 输入源
5. 内部伪代码：
   - step1: logger 改写到 ProblemMemory。
   - step2: raw_log_reader 仅解析 runs 路径。
   - step3: worklog_builder 仅消费 runs 事件流。
6. 修改文件：
   - `src/utils/logger.py`
   - `src/utils/raw_log_reader.py`
   - `src/utils/worklog_builder.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_infrastructure.py -k log_path -q`
8. 完成标准：新运行日志仅出现在 runs 路径。

---

### Phase B：协议与解析

#### B20 多标签解析与引用解析

1. 对齐依据：`architecture.md` 1.4；`idea.md` 1.2。
2. 借鉴来源：EvoScientist（协议字段独立解析）。
3. 目的/作用：支持多 lemma、verified_lemmas、cite、citation_review。
4. 输入输出：
   - 输入：LLM 文本输出
   - 输出：`lemmas[]`, `verified_lemmas[]`, `citations[]`, `citation_review`
5. 内部伪代码：
   - step1: `extract_xml_tags(tag)` 用 `finditer` 抓多块。
   - step2: 解析 `<lemma>` 和 `<verified_lemmas>`。
   - step3: 解析 `[cite:path]` 与 `<citation_review>`。
6. 修改文件：
   - `src/utils/parser.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_parser_contract.py -q`
8. 完成标准：混合样本可解析，坏块走可诊断错误。

#### B21 Prompt 合约固化

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2。
2. 借鉴来源：EvoScientist（提示词驱动的结构化动作）。
3. 目的/作用：确保 Generator/Verifier/Reviser 输出满足机器可解析契约。
4. 输入输出：
   - 输入：problem + context
   - 输出：包含必填标签的结构化文本
5. 内部伪代码：
   - step1: 定义必填标签清单。
   - step2: 缺失标签触发一次格式修复重试。
   - step3: 连续失败写 parse_error。
6. 修改文件：
   - `config/prompts.yaml`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_pipeline_contracts.py -k prompt -q`
8. 完成标准：缺标签时有稳定失败语义，不出现静默成功。

#### B22 解析错误分类与路由

1. 对齐依据：`architecture.md` 1.4；`idea.md` 2.2。
2. 借鉴来源：EvoScientist（异常结构化）。
3. 目的/作用：解析失败变成路由可消费状态，而非随机异常中断。
4. 输入输出：
   - 输入：原始文本、解析异常
   - 输出：`parse_error_code`, `failure_reason`
5. 内部伪代码：
   - step1: classify 为 invalid_verdict/malformed_tag/missing_solution。
   - step2: orchestrator 根据错误码决定重试或切换节点。
6. 修改文件：
   - `src/utils/parser.py`
   - `src/core/orchestrator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_orchestrator_route.py -k parse_error -q`
8. 完成标准：解析失败路径均可重现并可追踪。

#### B23 分层读取工具

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.1、3.1。
2. 借鉴来源：AutoResearchClaw（分层内容消费）。
3. 目的/作用：Agent 按需加载 Layer2/Layer3，避免长文本一次性注入。
4. 输入输出：
   - 输入：`path: str`, `layer: int`
   - 输出：层文本或结构化路径错误
5. 内部伪代码：
   - step1: 校验 path 必须位于 runs/{problem_id}/artifact 内。
   - step2: 解析三层 markdown。
   - step3: 返回目标层内容。
6. 修改文件：
   - 新建 `src/tools/artifact_reader.py`
   - `src/tools/registry.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_parser_contract.py -k layer_reader -q`
8. 完成标准：合法路径可读，路径穿越被拒绝。

---

### Phase C：Agent 化与调度主链替换

#### C30 BaseAgent 运行时骨架

1. 对齐依据：`architecture.md` 1.6.1；`idea.md` 2.2。
2. 借鉴来源：EvoScientist（动作循环与限次）。
3. 目的/作用：统一 Agent 的阶段记忆、工具循环与上限控制。
4. 输入输出：
   - 输入：`payload`, `tools`, `max_tool_rounds`
   - 输出：阶段最终文本
5. 内部伪代码：
   - step1: `reset_stage_memory()`。
   - step2: while rounds < max: chat -> tool_call -> append_result。
   - step3: 无 tool_call 返回最终文本。
6. 修改文件：
   - 新建 `src/agents/base.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_searcher_dedup.py -k base_agent -q`
8. 完成标准：阶段内记忆可见、阶段间记忆清空、超限可终止。

#### C31 GeneratorAgent 替换旧生成函数

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2、2.2。
2. 借鉴来源：EvoScientist（Agent 主循环）。
3. 目的/作用：从函数式生成切换为 Agent 对象执行。
4. 输入输出：
   - 输入：problem、layer1 summaries、error lessons
   - 输出：含 `<solution>` 和 `<lemma>` 的结构化文本
5. 内部伪代码：
   - step1: 拼接上下文。
   - step2: 调用 BaseAgent.run。
   - step3: 校验 lemma 与 cite 约束。
6. 修改文件：
   - 新建 `src/agents/generator.py`
   - `src/core/orchestrator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_pipeline_contracts.py -k generator -q`
8. 完成标准：旧 `call_generator` 不再是主入口。

#### C32 VerifierAgent 替换旧验证函数

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2。
2. 借鉴来源：EvoScientist + AutoResearchClaw（结构化输出 + 审查链）。
3. 目的/作用：输出统一 verdict/verification/verified_lemmas/citation_review。
4. 输入输出：
   - 输入：problem + solution
   - 输出：结构化验证文本
5. 内部伪代码：
   - step1: 主链验证。
   - step2: `<lemma>` 验证与筛选。
   - step3: 有 cite 则触发 CitationReviewer。
   - step4: 生成统一输出标签。
6. 修改文件：
   - 新建 `src/agents/verifier.py`
   - `src/core/orchestrator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_pipeline_contracts.py -k verifier -q`
8. 完成标准：`verified_lemmas` 可被落盘且路由稳定。

#### C33 ReviserAgent 替换旧修订函数

1. 对齐依据：`architecture.md` 1.2；`idea.md` 2.2。
2. 借鉴来源：EvoScientist（有状态阶段执行）。
3. 目的/作用：定点修补，不整稿重写。
4. 输入输出：
   - 输入：problem、previous_solution、verification_report
   - 输出：修订后的 `<solution>`
5. 内部伪代码：
   - step1: 标记问题段。
   - step2: 保留正确段。
   - step3: 仅替换缺陷段。
6. 修改文件：
   - 新建 `src/agents/reviser.py`
   - `src/core/orchestrator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_pipeline_contracts.py -k reviser -q`
8. 完成标准：修订范围可控，正确段不回归。

#### C34 调度主链替换（去掉旧适配主链）

1. 对齐依据：`architecture.md` 1.3；`idea.md` 2.2。
2. 借鉴来源：EvoScientist（编排与执行解耦）。
3. 目的/作用：让 Orchestrator 直接调用 Agent 对象，旧 pipeline 不再作为主执行链。
4. 输入输出：
   - 输入：problem_id、problem_text
   - 输出：final state + final output
5. 内部伪代码：
   - step1: generator.run -> parser。
   - step2: verifier.run -> route。
   - step3: reviser 或 generator 循环。
   - step4: finalizer 收敛输出。
6. 修改文件：
   - `src/core/orchestrator.py`
   - `src/core/agent.py`
   - `src/core/pipeline.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_orchestrator_route.py -q`
8. 完成标准：主流程不再依赖 `_PipelineAdapter`。

---

### Phase D：Searcher 与 CitationReviewer 子代理

#### D40 SearcherAgent 全链路实现

1. 对齐依据：`architecture.md` 1.2、1.6.2；`idea.md` 1.2、2.2。
2. 借鉴来源：AutoResearchClaw（稳定检索链）。
3. 目的/作用：形成可复用检索子代理，输出三层 papers artifact。
4. 输入输出：
   - 输入：`query` 或 `query_bundle`
   - 输出：`papers[]`（layer1/layer2/layer3 + source meta）
5. 内部伪代码：
   - step1: expand_queries。
   - step2: multi_source_search。
   - step3: dedup(doi > arxiv_id > normalized_title)。
   - step4: extract + summarize + add_paper。
6. 修改文件：
   - 新建 `src/agents/searcher.py`
   - 新建 `src/tools/search.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_searcher_dedup.py -q`
8. 完成标准：重复文献不重复落盘，输出结构稳定。

#### D41 Generator -> Searcher 桥接工具

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2。
2. 借鉴来源：EvoScientist（主 Agent 委派子代理）。
3. 目的/作用：让 Generator 在工具循环中按需调用 Searcher。
4. 输入输出：
   - 输入：`query: str`
   - 输出：检索摘要 + artifact 路径
5. 内部伪代码：
   - step1: registry 注册 `call_searcher` schema。
   - step2: 调用 SearcherAgent。
   - step3: 返回简短结果供下一轮推理。
6. 修改文件：
   - `src/tools/registry.py`
   - `src/agents/generator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_stage_integration.py -k generator_searcher -q`
8. 完成标准：Generator 能触发 Searcher 并继续完成阶段输出。

#### D42 CitationReviewer 子代理实现与按需触发

1. 对齐依据：`architecture.md` 1.2、1.6.2；`idea.md` 1.2。
2. 借鉴来源：AutoResearchClaw（级联引用核验）。
3. 目的/作用：将 citation 审查纳入标准输出链。
4. 输入输出：
   - 输入：`cites[]`, `claim_spans[]`
   - 输出：`citation_review {summary, items, fail_count}`
5. 内部伪代码：
   - step1: 每条 cite 做 path/meta/claim 一致性检查。
   - step2: 汇总通过率与失败项。
   - step3: 回传 Verifier 输出。
6. 修改文件：
   - 新建 `src/agents/citation_reviewer.py`
   - `src/agents/verifier.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_citation_review.py -q`
8. 完成标准：有 cite 必有 citation_review；无 cite 不触发审查。

#### D43 软门控路由落地

1. 对齐依据：`architecture.md` 3.5；`idea.md` 1.1。
2. 借鉴来源：AutoResearchClaw（审查结果与主流程解耦）。
3. 目的/作用：审查失败只记 warning，不阻断主流程。
4. 输入输出：
   - 输入：`citation_review.fail_count`
   - 输出：warning 事件 + final warning 摘要
5. 内部伪代码：
   - step1: if fail_count > 0 append warning event。
   - step2: 路由不改为 fail。
   - step3: finalizer 汇总 warning。
6. 修改文件：
   - `src/core/orchestrator.py`
   - `src/core/finalizer.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_orchestrator_route.py -k citation -q`
8. 完成标准：warning 可追踪且流程可收敛。

---

### Phase E：Finalizer 与引用导出

#### E50 引用构建器

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2。
2. 借鉴来源：AutoResearchClaw（引用核验与导出链）。
3. 目的/作用：把 `[cite:path]` 转为 `[1][2]` 并生成 references。
4. 输入输出：
   - 输入：`solution_text`, `problem_memory`
   - 输出：`converted_text`, `references[]`, `missing_warnings[]`
5. 内部伪代码：
   - step1: 按首次出现编号。
   - step2: 用 Layer3 元信息生成引用条目。
   - step3: 缺失路径写 warning。
6. 修改文件：
   - 新建 `src/utils/reference_builder.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_finalizer_reference.py -k builder -q`
8. 完成标准：重复 cite 编号一致。

#### E51 Finalizer 模板重构

1. 对齐依据：`architecture.md` 1.3（final 阶段）；`idea.md` 1.2（References）。
2. 借鉴来源：AutoResearchClaw（最终产物组织）。
3. 目的/作用：统一输出正文 + References + Citation Warnings。
4. 输入输出：
   - 输入：`solution`, `references`, `warning_summary`, `status`
   - 输出：统一 final markdown
5. 内部伪代码：
   - step1: 替换 cite。
   - step2: 追加 References。
   - step3: 追加 Citation Warnings。
6. 修改文件：
   - `src/core/finalizer.py`
   - `src/core/orchestrator.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_finalizer_reference.py -k final_output -q`
8. 完成标准：success/partial/fail 模板统一。

#### E52 BibTeX 导出

1. 对齐依据：`architecture.md` 1.2；`idea.md` 1.2。
2. 借鉴来源：AutoResearchClaw（BibTeX 清洗与输出）。
3. 目的/作用：输出可复用的 `citations.bib`。
4. 输入输出：
   - 输入：`references[]`
   - 输出：`artifact/citations.bib`
5. 内部伪代码：
   - step1: reference -> bib entry。
   - step2: 缺失字段占位。
   - step3: 保存到 ProblemMemory。
6. 修改文件：
   - `src/utils/reference_builder.py`
   - `src/memory/problem_memory.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_finalizer_reference.py -k bibtex -q`
8. 完成标准：bib 文件可生成且格式可解析。

#### E53 旧 final 分支清理

1. 对齐依据：`architecture.md` 3.1；反馈要求（替代后不保留回退）。
2. 借鉴来源：EvoScientist（保持主链清晰，减少分支漂移）。
3. 目的/作用：清理旧输出分支，避免双逻辑并存。
4. 输入输出：
   - 输入：统一 final 数据结构
   - 输出：单一路径 final 结果
5. 内部伪代码：
   - step1: 删除旧模板分支。
   - step2: 统一调用 reference_builder。
   - step3: 写最终输出。
6. 修改文件：
   - `src/core/finalizer.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_finalizer_reference.py -q`
8. 完成标准：final 输出只有一条主链。

---

### Phase F：测试、评测、E2E 与最终门禁

#### F60 单元测试（memory/parser/protocol）

1. 对齐依据：`architecture.md` 3.9；`idea.md` 可验证要求。
2. 借鉴来源：ResearchClaw（状态层可测试性）。
3. 目的/作用：确保底座与协议稳定。
4. 输入输出：
   - 输入：固定样本与临时目录
   - 输出：测试断言
5. 内部伪代码：
   - step1: memory 写读。
   - step2: parser 多标签解析。
   - step3: 协议异常分支。
6. 修改文件：
   - 新建 `tests/test_problem_memory.py`
   - 新建 `tests/test_parser_contract.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_problem_memory.py tests/test_parser_contract.py -q`
8. 完成标准：两类单测全绿。

#### F61 集成测试（orchestrator/agent/subagent）

1. 对齐依据：`architecture.md` 1.3；`idea.md` 2.2。
2. 借鉴来源：EvoScientist + AutoResearchClaw（编排+子代理链路）。
3. 目的/作用：验证主路由与子代理调用链联通。
4. 输入输出：
   - 输入：mock llm/tool 输出
   - 输出：路由断言与落盘副作用断言
5. 内部伪代码：
   - step1: 覆盖 CORRECT/MINOR/CRITICAL。
   - step2: 覆盖 generator->searcher。
   - step3: 覆盖 verifier->citation_reviewer。
6. 修改文件：
   - 新建 `tests/test_orchestrator_route.py`
   - 新建 `tests/test_stage_integration.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest tests/test_orchestrator_route.py tests/test_stage_integration.py -q`
8. 完成标准：主路由与子链路全通过。

#### F62 评测脚本切换到 runs

1. 对齐依据：`architecture.md` 3.7 P2；反馈要求（不保留回退）。
2. 借鉴来源：AutoResearchClaw（可复现实验产物组织）。
3. 目的/作用：让 benchmark 只依赖新目录结构。
4. 输入输出：
   - 输入：dataset/count/max_turns
   - 输出：summary json + 新指标
5. 内部伪代码：
   - step1: resolve_run_log_path 仅解析 runs。
   - step2: 统计 lemma_accept_rate/citation_warning_rate。
   - step3: 输出结果文件。
6. 修改文件：
   - `scripts/run_imobench.py`
   - `scripts/run_proofbench_advanced.py`
   - `src/utils/raw_log_reader.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe scripts/run_imobench.py --dataset answerbench --count 2 --max-turns 2`
8. 完成标准：脚本不再读取 `data/logs`。

#### F63 实时 E2E 监控对齐 runs

1. 对齐依据：`architecture.md` 3.9；`idea.md` 调试可追踪要求。
2. 借鉴来源：ResearchClaw（可观测视图与状态跟踪）。
3. 目的/作用：实时监控直接追踪 runs 事件流。
4. 输入输出：
   - 输入：`input_file`, `max_turns`
   - 输出：tracking markdown（final_status/warning_count/bug_count）
5. 内部伪代码：
   - step1: 启动 main。
   - step2: 轮询 runs 下 jsonl。
   - step3: 汇总 warning 与 bug。
6. 修改文件：
   - `tests/realtime_e2e_monitor.py`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe tests/realtime_e2e_monitor.py --input tasks_v1.md --max-turns 2`
8. 完成标准：tracking 文件包含最终状态与信号统计。

#### F64 最终门禁与缺陷闭环

1. 对齐依据：`architecture.md` 3.9；`idea.md` 闭环修复要求。
2. 借鉴来源：三参考项目共同的“可复现+可回溯”原则。
3. 目的/作用：确认系统可运行、可回归、可复现。
4. 输入输出：
   - 输入：全量测试 + 小规模 benchmark
   - 输出：门禁结果 + 修复记录
5. 内部伪代码：
   - step1: 执行 pytest 全量。
   - step2: 执行 proofbench 小样本。
   - step3: 更新 `docs/e2e_fix_log.md`。
6. 修改文件：
   - 新建 `docs/e2e_fix_log.md`
7. 验收命令：
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe -m pytest -q`
   - `d:/Project/AletheiaReproduction/.venv/Scripts/python.exe scripts/run_imobench.py --dataset proofbench --count 2 --max-turns 2`
8. 完成标准：测试全绿，门禁清单全部通过。

---

## 4. 执行纪律

1. 每完成一个任务，必须同时提交对应测试。
2. 未通过当前任务验收，不得进入下一任务。
3. 每次提交必须写清输入、输出、失败场景、边界条件。
4. 禁止继续引入“回退/降级路径”并行维持旧实现。
