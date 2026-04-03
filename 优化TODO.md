# 优化TODO（基于代码审计 + 一次真实E2E）

## 审计结论总览

- P0（必须优先处理）
  - 检索子链路在生产路径默认无数据源，`call_searcher` 可执行但常态返回空结果，属于“接口存在、能力未接通”。
  - 工具异常未统一结构化，仍有字符串错误返回，缺少可恢复中间件（重试/分类/可观测字段）。
  - Verifier 对基础公理类陈述存在“过严误报”风险，导致 trivial 题也可能进入 PARTIAL。
- P1（高价值）
  - `idea.md` 中“引理写入 Source 字段、拒绝仅概要证明”等约束未形成可执行校验。
  - 引用审查仅做文件/词重叠/元信息检查，未落地 DOI->OpenAlex->arXiv->title 级联真实性验证。
- P2（中期治理）
  - 文档与实现存在错位（README 仍描述旧工具与旧结构）。
  - 若干模块仍有“接口保留但真实能力弱绑定”的技术债，需要统一注入策略与回归用例。

---

## 真实E2E记录（本次）

- 运行命令：
  - `.venv/Scripts/python.exe main.py --problem "Prove that for all integers n, n(n+1) is even." --max-turns 1 --no-generate-worklog`
- 产物：
  - `runs/inline_20260402_170041/history.jsonl`
  - `runs/inline_20260402_170041/state.json`
  - `runs/inline_20260402_170041/artifact/final_output.md`
- 观测到的 warning/bug：
  - W1: Verifier 将“未显式说明奇偶完备性依据”判为 `MINOR_FLAW`，最终 `PARTIAL_PROGRESS`。
  - W2: FINAL 输出中 `<undone>无</undone>`，但 run status 仍为 `PARTIAL_PROGRESS`，状态语义与文本存在张力。
- 初步归因：
  - R1: Verifier Phase3 规则对“基础公理/常识”缺少白名单或软化准则，易触发误报。
  - R2: `_finalize_exhausted` 只基于最后 verdict 路由，不校验 FINAL 的 done/undone 语义一致性。

---

## P0（本周必须完成）

### P0-1 接通 Searcher 真实数据源（非测试桩）

- 问题
  - 当前 `configure_searcher_sources` 仅被测试调用，生产路径未注入 source handlers。
- 目标
  - 在主运行路径中注入至少 1-2 个可用检索源（本地/HTTP），`call_searcher` 默认可返回真实候选论文。
- 验收标准
  - 新增集成测试：不 mock `configure_searcher_sources` 也能得到 `paper_count > 0`（在可控假源或录制数据下）。
  - 真实运行中 `history.jsonl` 的 `VERIFIER/GENERATOR` 可观测到非空 papers 路径。
  - 失败时返回结构化错误（见 P0-2），而非静默空集合。

### P0-2 工具异常中间件结构化（对齐 EvoScientist 思路）

- 问题
  - `execute_tool` 对未知工具/运行异常返回字符串 `[TOOL ERROR] ...`，缺少统一 machine-readable 结构。
- 目标
  - 所有工具调用返回统一 JSON 包络：`status/error_code/retryable/message/detail/tool/trace_id`。
  - 引入有界重试与错误分类（timeout/network/validation/runtime）。
- 验收标准
  - `chat_with_tools` trace 中可稳定解析每次工具失败类型。
  - Orchestrator/Verifier 可基于 `retryable` 做恢复或降级决策。
  - 新增测试覆盖：未知工具、参数错误、运行时异常、超时。

### P0-3 Verifier 误报抑制（基础事实软化）

- 问题
  - 简单题被判 `MINOR_FLAW`，导致整体误降级到 PARTIAL。
- 目标
  - 在 verifier rubric 增加“基础事实可隐式接受”的边界策略，避免无意义扣分。
- 验收标准
  - 用当前 E2E 题复跑，默认应达到 `CORRECT` 或至少不触发“形式主义误报”。
  - AnswerBench/ProofBench 采样回归中，`verifier_false_negative` 比例下降。

### P0-4 FINAL 状态一致性守卫

- 问题
  - 允许出现“文本上已完整，状态仍 PARTIAL”的冲突。
- 目标
  - 在 `_finalize_exhausted` 增加一致性检查（例如 `<undone>` 为空/无缺口时可升级或触发二次核验）。
- 验收标准
  - 新增回归测试覆盖上述冲突场景。
  - manifest 中状态与 final_output 语义一致。

---

## P1（高优先级改进）

### P1-1 引理入库协议硬校验

- 问题
  - `verified_lemmas` 已解析入库，但未强校验 Source 字段、完整证明粒度。
- 目标
  - 定义引理最小协议（`Source`、证明主体、可追溯引用），不满足则拒绝入库并记录 error artifact。
- 验收标准
  - 对“仅摘要引理”输入返回拒绝并给出 machine-readable 原因。
  - `artifact/lemmas/*.md` 均满足统一三层模板。

### P1-2 引用真实性级联核验增强

- 问题
  - 目前 CitationReviewer 仅本地一致性审查，未做外部标识级联验证。
- 目标
  - 增加 DOI/OpenAlex/arXiv/title 级联核验策略（可配置开关，默认 soft gate）。
- 验收标准
  - 新增测试覆盖：DOI 命中、arXiv 命中、title fallback、全失败。
  - `citation_review` 结构包含级联链路证据与置信度来源。

### P1-3 Searcher 去重与可追溯性增强

- 问题
  - 已有 DOI/arXiv/title 去重，但缺少“已存在 artifact 反查 + 命中原因”输出。
- 目标
  - 返回每篇论文的去重命中依据与来源合并轨迹。
- 验收标准
  - `papers[]` 新增 `dedup_key`、`merged_from` 等字段。
  - 重复检索时可明确看到“复用已有文档”而非重写。

---

## P2（工程治理）

### P2-1 README 与代码对齐

- 问题
  - README 仍描述已不存在或未接通的工具/结构，影响协作判断。
- 目标
  - 更新架构图、工具表、目录树，明确当前真实能力与限制。
- 验收标准
  - 文档中所有列举文件在仓库中实际存在。
  - 运行示例命令可直接复现。

### P2-2 审计可观测性统一

- 目标
  - 统一 run 级指标导出（tool_error_rate、citation_fail_rate、verifier_false_negative 等），减少手工排查。
- 验收标准
  - `scripts/run_imobench.py` 输出统一质量指标并写入 JSON summary。

### P2-3 测试分层补强

- 目标
  - 将“单元可过但真实链路空转”的风险纳入 CI。
- 验收标准
  - 新增 smoke E2E（真实 main 路径 + 最小可用检索源）与契约测试（tool error envelope）。

---

## 建议实施顺序

1. P0-2 工具异常中间件（为后续所有链路提供稳定错误语义）
2. P0-1 Searcher 真实数据源接通（先打通真实能力）
3. P0-3 + P0-4 Verifier/FINAL 状态语义修复（降低误报和错分）
4. P1 系列协议增强（引理与引用真实性）
5. P2 文档与治理
