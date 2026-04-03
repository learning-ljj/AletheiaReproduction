# Retrieval & Citation Full-Chain TODO

## 1. 目标与边界

目标：把当前 Searcher/CitationReviewer 的轻量链路升级为接近 architecture.md 的全量能力，包括“查询扩展 -> 多源检索 -> 去重重排 -> PDF 抽取 -> claim 萃取 -> 引用级联核验 -> FINAL 引用产物打包”。

边界：
- 当前轮只完成任务设计、参考迁移映射和避坑清单。
- 当前代码已落地的能力（query expansion、multi-source、dedup、candidate claims、citation evidence）继续保留。

## 2. 子任务拆解（按依赖顺序）

### T1. 统一检索源适配层

实现项：
- 定义统一 SourceAdapter 协议：search(query, limit) -> list[PaperRecord]。
- 为 Semantic Scholar / arXiv / Wikipedia 编写适配器。
- 增加 provider 熔断与失败降级（单源失败不阻塞全链路）。

落位：
- src/tools/search_sources.py
- src/agents/searcher.py
- config/settings.yaml retrieval.providers

验收：
- 多源可并行或串行执行，任一源失败时仍可返回其它源结果。

### T2. PDF 拉取与正文抽取

实现项：
- 根据 DOI/arXiv URL 获取 PDF 地址。
- 本地缓存 PDF（按 paper_id 命名）。
- 提取正文分段并保留页码定位。

落位：
- src/tools/pdf_fetcher.py
- src/tools/pdf_extractor.py
- runs/{problem_id}/artifact/papers/raw/

验收：
- 任意论文至少输出正文片段列表，含 section/page 元信息。

### T3. 结构化 claim/evidence 萃取

实现项：
- 对正文执行 claim 句子级候选抽取（定理/引理/结论句）。
- 为每条 claim 绑定 evidence spans（页码、段落、原文片段）。
- 输出 claim_graph.json（可选视图，不阻塞主流程）。

落位：
- src/agents/searcher.py
- src/tools/claim_extractor.py
- runs/{problem_id}/artifact/claim_graph.json

验收：
- 每个 paper 文件 Layer2 至少包含 claim 列表，Layer3 包含 evidence 定位信息。

### T4. CitationReviewer 级联核验（硬能力）

实现项：
- 路径存在性 + 句子级 claim-source match。
- 元信息一致性（title/authors/doi/arxiv/url）。
- DOI -> OpenAlex -> arXiv -> title fallback 级联查询。
- 输出 confidence 与 suggested_action。

落位：
- src/agents/citation_reviewer.py
- src/tools/citation_providers.py
- config/settings.yaml verifier.citation_review

验收：
- <citation_review> 包含每条 cite 的核验证据与置信度。

### T5. FINAL 产物打包

实现项：
- 生成 artifact/manifest.json（solution、references、citation_review、errors）。
- 生成 citations.bib 并支持缺失条目告警。
- 输出 references 的编号稳定性测试。

落位：
- src/core/orchestrator.py
- src/utils/parsing/reference_builder.py
- src/core/finalizer.py

验收：
- final_output + references + bib + manifest 四件套一致可追溯。

## 3. 参考项目迁移映射

### 3.1 AutoResearchClaw

参考文件：
- 参考AutoResearchClaw/stable_literature_search.py
- 参考AutoResearchClaw/citation_review.py
- 参考AutoResearchClaw/latex_pipeline.py

迁移策略：
- 复用“检索链顺序 + 去重键优先级 + 引文级联核验”思想。
- 不直接复制实现，优先改造成当前项目的 SourceAdapter/CitationProvider 接口。

### 3.2 EvoScientist

参考文件：
- 参考evoscientist/portable_agent_core/agent_framework.py
- 参考evoscientist/portable_agent_core/resilience.py

迁移策略：
- 借鉴受保护工具调用与有界重试。
- 保持 Orchestrator 只做阶段路由，避免回退成“大一统控制器”。

### 3.3 ResearchClaw

参考文件：
- 参考ResearchClaw/research_workflow_engine.py

迁移策略：
- 借鉴 typed state + 快照持久化。
- claim/evidence 作为扩展视图，不阻塞主链收敛。

## 4. 迁移风险与避坑

1. 路径与编码
- 风险：Windows 路径反斜杠进入 JSON/XML 导致解析失败。
- 对策：统一 as_posix + JSON 函数替换写回，不做字符串模板替换。

2. 工具重试副作用
- 风险：有副作用工具重试导致重复落盘。
- 对策：写入前做幂等检查，文件内容一致则不重复写。

3. 引用核验误判
- 风险：只做 substring 匹配会误判。
- 对策：句子级 span + token overlap + metadata 级联联合判定。

4. 检索噪声膨胀
- 风险：查询扩展过度，结果质量下降。
- 对策：限制扩展上限并按相关度/来源可信度排序。

5. 协议漂移
- 风险：verifier/citation 输出字段变化导致解析失败。
- 对策：固定输出 schema 并用 contract tests 锁定。

## 5. 里程碑验收建议

M1：检索源适配 + PDF 抽取可跑通。
M2：claim/evidence 与 citation 级联核验打通。
M3：FINAL 打包与评测脚本完全消费新产物。

建议命令：
- pytest tests/test_searcher_dedup.py tests/test_citation_review.py tests/test_finalizer_reference.py -q
- pytest tests/test_stage_integration.py tests/test_orchestrator_route.py -q
