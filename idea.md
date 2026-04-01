## 当前项目框架

当前框架的核心是 **“单向流水线 + 中心化状态记录”**。它并不是一个真正的多智能体（Multi-Agent）系统，而是一个由 Orchestrator 驱动的**函数调用链**。

1. **调度层 (`orchestrator.py` & `agent.py`)**
    - **硬编码的线性流**：逻辑写死在 `run()` 方法的 `for` 循环中：Generator -> Verifier -> (Reviser | Generator) -> ... -> FINAL。
    - **无自主决策**：Orchestrator 本身不包含 LLM，它只是一个死板的路由，完全依赖 Verifier 输出的枚举值 (`CORRECT`, `MINOR_FLAW`, `CRITICAL_FLAW`) 决定下一步走哪。
        
2. **执行层 (`pipeline.py`)**
    - **Generator / Reviser 是“瞎子”**：它们被设计为纯函数，只能调用 `llm_client.chat(thinking=True)`。它们无法使用工具，只能凭空“想”出解答或基于错误报告进行修改。
    - **工具调用仅限 Verifier**：其他节点无法调用工具。Verifier 分为三个 Phase，只有它能调用 `llm_client.chat_with_tools` 来执行 Python 验证算术或搜索 Wikipedia/arXiv。
    - **最终输出简单**：仅返回 solution 文本或失败原因，没有参考文献、已验证中间结果等结构化内容。
        
3. **状态与记忆 (`state.py` & `logger.py`)**
    - **流水账式记忆**：`ProofState` 的核心是 `history: list[VerificationLog]`。它记录的是“发生过什么”（Turn 1 说了什么，Turn 2 报错了什么），而**不是“沉淀了什么知识”**。
    - **上下文堆叠**：记忆传递完全依赖把上一步的输出（如 `verification_report`）拼接到下一步的 Prompt 中。
    - **无状态 Agent**：Generator/Reviser/Verifier 每次调用都是独立的，没有内部记忆，完全依赖传入参数。只是简单的状态机。
    - **有限的共享记忆**：只有 `ProofState.history` 记录了历史只保存当前解题过程的 `current_proof`、历史日志、迭代次数，不包含任何外部知识（论文、引理等）的持久化存储。 Generator/Reviser 并未主动利用它（除了 lesson 字符串）。
        
4. **工具与文献 (`registry.py`)**
    - **粗暴的文献处理**：`read_arxiv_latex` 直接下载源码，通过简单的 `max_chars=6000` 截断。这会把大量 LaTeX 导言区（预处理、包引用）塞给大模型，真正的定理和证明可能根本没截取到。

# 目标功能（需要修改代码实现）

## 1. 状态层

### 1.1 状态与记忆 - 扩展状态模型（ProofState → ProblemMemory）

**你已明确的：**
- Generator、Reviser、Verifier 从“函数”升级为“Agent实体”。它们拥有独立的思考域，且全员均可调用工具（查阅文献、写代码试错），拥有自己私有的 memory 和共享的 memory。
  - **Agent memory**：每个 Agent 内部试错、推导的 Scratchpad。
  - **Problem Memory**：将 ProblemMemory 作为**一个单独的类**。存放公共资产，如ProofState、“已提取清洗的文献”、“验证通过的局部引理”、“死胡同路径记录”。同时负责读写维护 artifact 文件
- 每个问题维护一个**状态对象Problem Memory**。状态对象需要包含：ProofState、检索并经过知识提取的论文内容以及对应的原文链接和pdf、已验证的中间结果、错误分析报告等。
- 中间结果（提取后的论文内容、引理、错误报告）以 markdown 文件形式存储在 `artifact/` 下，`ProblemMemory` 中仅存“概述”（YAML frontmatter），实现**分层暴露**。分层暴露中的“概述”是searcher agent返回结果时就有的（用 LLM）
- “已验证的中间结果（引理）”的来源：Generator 解题过程中提出的引理，经 Verifier 验证。 Verifier 优先验证引理
  - 注意：generator的最终输出需要要求Generator在输出中明确标记<lemma>...</lemma>。这样的话 generator的思维习惯也得修改了（通过prompt），比如：先分类讨论，先证明需要用到的小结论（引理），再进行最后的分步解答
  - verifier同时也得修改一下，通过输出 `<verified_lemmas>` 标签，由 Orchestrator 解析后调用 `ProblemMemory.add_lemma()`

**进一步明确：**
1. **ProblemMemory 的定位**
   - 是作为一个全局单例（所有 Agent 共享），将 `ProofState` 修改后完全合并进 `ProblemMemory`，`ProofState`负责状态保存（只保存 `current_proof`、`iteration_count`、`status` 等关键字段），`ProblemMemory`中还有其他类方法的实现读写artifact文件夹下的内容
   - ProofState 负责的状态保存也需要写入 artifact 下的某个文件，提供 save_state() 和 load_state() 方法，并且与引理、论文等文件区分存储。
   - 原先的ProofState.history（原有的 `logs/{problem_id}.jsonl` 功能（后者由 `logger.py` 写入 raw 事件））单独存储于与artifact同级的目录中，针对每一个问题的ProblemMemory管理的文件架构如下：
     ```
     {problem_id}/
       history.jsonl             # 废弃原有的logs/目录下的.jsonl文件，将原有逻辑转为统一使用history.jsonl存放 raw 事件，由 `Orchestrator` 负责写入，保持一致性。
       state.json                 # 保留ProofState的关键字段
       artifact/
         lemmas/
             001.md                 # 引理1
             002.md
           papers/
             arXiv_2501.12345.md    # 论文提取内容
             arXiv_2501.67890.md
           errors/
             001.md                 # 错误分析记录
           citations.bib            # 最终引用的bib文件（可选）
     ```
   - ProblemMemory 只存“概述”，即定理/引理的使用条件和结论， Agent 自行决定选择，如果要使用该定理/引理，再读取artifact中的完整信息
   - ProblemMemory 是“每个问题一个实例”，由 Orchestrator 在启动时创建并传递给 Agent。需要设计为按 problem_id 隔离目录。
   - Verifier负责验证中间结果， Orchestrator 负责解析验证后结果并用 ProblemMemory写入artifact。
2. **“分层暴露”的具体粒度**
   - 对于‘提取后的论文内容’和验证后的‘generator生成的引理’在artifact中存储的文本格式：
     - 第一层是定理/引理的使用条件和结论（YAML frontmatter）；
     - 第二层是定理的原文对应完整证明过程内容。（如果是generator生成的引理，给出完整证明推导过程）
     - 第三层是提取的定理/引理来源（对应的论文标题、作者等引用元信息、对应于原文的所在位置）（如果是generator生成的引理，标明即可）；
   - 这种分层需要 Agent 在 Prompt 中动态插入，由 LLM 直接通过工具调用来读取解析更深层的加载请求
   - 读取工具的参数只接收文件路径。如果 Agent 想要读取某个引理的“完整证明”（第二层），工具应该只返回第二层部分。其他层的读取也需要实现类似的工具。
   - 不设计设置引理数量限制。
3. **错误分析报告的形式与来源**
   - 错误报告由 Verifier 在 Phase 3 生成是纯文本
4. **状态模型与现有 ProofState 的关系**
   - 新建一个 `ProblemMemory` 类，包含 `ProofState` 类的功能，同时负责管理读写artifact文件夹中的中间结果
   - Orchestrator 将 ProblemMemory 传递给 Agent：每次用户输入前，初始化时准备好；运行时调用ProblemMemory类的方法进行上下文管理

### 1.2 状态与记忆 - 实现问题级记忆的读写与分层暴露

**你已明确的：**
- 要求 Generator 在输出中明确标记 `<lemma>...</lemma>`，表示提出需要验证的引理。
- 利用子 agent（如 Searcher）检索、筛选、提取文献，返回 markdown 文件，开头包含简要概述，实现分层暴露（类似“Skill”的三层按需加载）。
- 分层暴露的实现方式：第一层（YAML frontmatter）启动时加载，第二层（正文）决定用时读取，第三层（引用文件）仅在引用时加载。

**进一步明确：**
1. **引理标记与验证流程**
   - Generator 输出中的`<solution>`可能包含多个 `<lemma>`标签，修改prompt，要求`<lemma>` 一定出现在 `<solution>` 内部最前面，且每个 `<lemma>` 独立成块。
   - 将引理作为解答的一部分，由主 Verifier 在验证整个解答时进行处理，如果某个引理成立，就将其输出，让Orchestrator调用 ProblemMemory的类方法写入artifact
   - 如果某个引理被误判为正确并存入 ProblemMemory，generator如果使用了引理，需要调取输出完整的证明过程（存在markdown的第二层），在verifier验证时会又一次验证
   - 如果验证失败，则分析对解答过程正确性的影响，把失败原因写入错误报告，verifier判断路由，转交给generator或reviser
   - 如果 Generator 提出了多个引理，部分成功部分失败，只把成功的存入 ProblemMemory
   - Verifier 在验证时只负责审查和问题暴露，如果Generator 在提出引理时并没有提供完整证明（只给了证明概要），则不进行补充证明、拒绝存储该引理，并将该问题在report中提出
   - Verifier 在验证整个解答时，同时处理 Generator 提出的 `<lemma>`。验证成功的引理使用 ProblemMemory的类方法写入artifact下的markdown文件；验证失败的引理，将失败原因作为“教训”放到错误报告中反馈给 Generator。Generator 提出的引理如果只给了证明概要，Verifier 不补充证明，直接拒绝并指出问题。
   - 路由逻辑是：CRITICAL_FLAW → Generator，MINOR_FLAW → Reviser。如果引理失败属于 CRITICAL（因为依赖错误的前提），应该走 Generator；如果只是证明细节不严谨但引理本身正确，可能走 Reviser。这个判断需要 Verifier 给出明确的 verdict，允许区分。
   - **引理写入**：让 Verifier 在 `<verified_lemmas>` 中完整复制 Generator 自证的 `<lemma>` 内容（包括 Proof ），而不仅仅是输出条件结论；如果Generator 在输出 `<lemma>` 时引用了artifact中的结论则注明引用路径，方便验证是否正确引用（包括引用位置和引用内容）；如果是自证的也注明 —— 要包含 `Source:` 字段
2. **子 agent 的角色与生命周期**
   - Searcher 是作为工具被 Generator 调用，作为一个独立 agent，负责检索、清洗论文，并返回 markdown 文件
   - 如果 Searcher 是独立 agent，Generator负责启动它，由 Generator 在需要时调用
   - 子 agent 的返回值（markdown 文件）的第一层暴露作为主 Agent （generator）的输入而被引用
3. **分层暴露的实现细节**
   - Agent 可以按需加载‘第二层’，由 LLM 决定使用定义的工具来实现读取。在 Prompt 中告知 Agent 如何发出加载请求，并且需要在 Agent 的调用循环中处理这种特殊输出（类似工具调用）
   - Agent 自己组装 Prompt 时去读取artifact中各引理的第一层
4. **记忆与分层暴露的耦合**
   - ProblemMemory 中存储的是“概述”，而分层暴露指的是在 Prompt 中动态注入。由 Orchestrator 来负责将“概述”统一注入到 Prompt。
5. **最终输出与引用追踪**
   - 当 Generator 调用 Searcher 并读取论文时，该论文的引用信息（arXiv ID, title, authors）应该被Searcher agent记录并存储到返回的 markdown 文件的第三层。
   - 段落级引用的插入：在有引用的每个段落后用 [1] 形式标注来源。要求 Generator 在生成解答时输出当前段落的信息来自哪个文件（定位到artifact即可，最后由Orchestrator中最后的final模块将artifact路径替换为所引用文件的第三层内容，即标准的正规引用格式），最终输出的 markdown 文件需要包含一个 ## References 章节，列出所有引用。需要按出现顺序编号，需要支持 BibTeX 导出。
   - 在Generator Prompt 中强制要求：**任何引用外部知识（论文、引理）时，必须紧跟着写出 `[cite:文件路径]`**，其中文件路径是 Searcher 返回的路径或引理文件路径。

- **重复/矛盾标注**：
  - **关于 ProblemMemory 是否为全局单例**：阶段1中“进一步明确”说“是作为一个全局单例（所有 Agent 共享）”，但同一条中又说“ProblemMemory 是‘每个问题一个实例’”。存在矛盾。需标明：前者指运行时单例？后者明确每个问题一个实例。建议采用“每个问题一个实例”的设计。
  - **关于 ProofState.history 存储位置**：阶段1中说“原先的ProofState.history...单独存储于与artifact同级的目录中”，但随后文件架构示例中写的是 `{problem_id}/history.jsonl`，并说“废弃原有的 `logs/` 目录，将原有逻辑转为统一使用 `{problem_id}/artifact/history.jsonl`”。这里不一致：先说是与artifact同级，后说放在artifact内部。矛盾。需标明：最终方案为 `{problem_id}/artifact/history.jsonl`（由 Orchestrator 写入）。

## 2. 执行层（agent编排）

### 2.1 状态与记忆 - 扩展状态模型（ProofState → ProblemMemory）

**你已明确的：**
- Verifier 优先验证引理
- verifier同时也得修改一下，通过输出 `<verified_lemmas>` 标签，由 Orchestrator 解析后调用 `ProblemMemory.add_lemma()`

**进一步明确：**
1. **ProblemMemory 的定位**
   - Verifier负责验证中间结果， Orchestrator 负责解析验证后结果并用 ProblemMemory写入artifact。
4. **状态模型与现有 ProofState 的关系**
   - Orchestrator 将 ProblemMemory 传递给 Agent：每次用户输入前，初始化时准备好；运行时调用ProblemMemory类的方法进行上下文管理

### 2.2 执行层 - 将三个节点函数重构为有状态的 Agent 对象

**你已明确的：**
- Agent 拥有**私有短期记忆**（仅当前阶段（如第二次generator）中产生的message，如已调用的工具以及执行结果），不保存跨阶段的历史（历史经验由 Orchestrator 通过 `verification_report` 传递）。
- 核心目标是重构后依旧能稳定运行。

**进一步明确：**
1. **Agent 的接口设计**
   - 每个 Agent 的 `run` 方法接收的，由proofstate包装好：
     - generator接收问题描述、当前artifact文件夹下的引理、之前总结的错误思路（同一道题目的历史经验）
     - reviser接收问题描述，verifier输出的错误报告，以及generator的完整解答
     - verifier接收问题描述和generator的完整解答
   - 保证 Agent 对状态是“只读”的，状态交由Orchestrator维护管理
2. **私有短期记忆的实现**
   - 所有agent都内含react模式，对于某个问题的单个运行阶段（比如第一轮generator），agent不断进行thinking、action的交错运行，这时候的多轮工具调用需要保留调用记录和结果，也需要保留对话记录。这一部分是属于单个agent的内部记忆，因此是agentMemory。
   - 但是在该阶段运行结束后，下一阶段（比如第一轮verifier，第二轮generator）就不会有上一阶段的memory。——Generator/Reviser/Verifier这三个agent的私有记忆只存在于当前阶段。
   - Agent 的私有短期记忆不在多个阶段之间保留。
   - 每个问题都新建 Agent 实例，那么问题结束后这些实例会被丢弃，但在解决各个问题的过程中的不同的阶段(比如第一次、第二次generator……)，重置其短期记忆，不保留任何历史信息，而非销毁实例。
   - 需要在 Agent 实例中显式保存 self.messages 作为短期记忆，并在每次新阶段开始时重新初始化。
3. **Agent 的初始化与依赖注入**
   - Generator/Reviser/Verifier 在每个问题运行时创建。在每个阶段当 Agent 完成工具调用并输出最终解答时，agent需要进行初始化，不保留任何历史信息。
   - 当 Generator/Reviser 获得工具调用能力后，需要像 Verifier 那样支持**多轮工具调用**（即一次 Agent 调用中可以多次使用工具）
   - 如果 Generator/Reviser 未来需要工具调用，工具列表应该配置（各个agent的工具/subagent列表不同），但同时需要引入硬性的动作截断机制，限制工具调用次数（例如每轮最多调用 5 次）。
   - 需要为每个 Agent 设置独立的 max_tool_rounds。
   - search-agent负责去重（避免重复检索同一篇论文）
   - Searcher 作为独立 subAgent，由 Generator 通过工具调用启动。Searcher 负责检索、清洗论文，返回 markdown 文件，并负责去重。Searcher 产出的结果写入 artifact。
   - Searcher 负责去重，避免重复检索同一篇论文。将当前问题的 ProblemMemory 维护的内容作为输入告诉Searcher agent，这样 Searcher 可以检查 ProblemMemory.papers 目录下是否已经存在该论文。
   - Searcher 返回的 markdown 文件应该遵循相同的三层结构：
     - 第一层 (YAML frontmatter)：定理/引理的使用条件和结论，简洁明了
     - 第二层：提取的定理/引理完整内容。
     - 第三层：引用的原文完整证明过程内容，arXiv ID, title, authors
4. **Orchestrator 的适配**
   - 当前 Orchestrator 直接调用 `pipeline.call_generator(...)`，改为 Agent 后，Orchestrator 需要调用 `agent.run(...)`。
   - 保持 Orchestrator 的简单循环，仍然由它决定何时调用哪个 Agent
   - 由 Orchestrator 解析verifier验证后的引理后调用 `ProblemMemory.add_lemma()`

### 2.3 状态与记忆 - 实现问题级记忆的读写与分层暴露

**你已明确的：**
- 利用子 agent（如 Searcher）检索、筛选、提取文献，返回 markdown 文件，开头包含简要概述，实现分层暴露（类似“Skill”的三层按需加载）。

**进一步明确：**
1. **引理标记与验证流程**
   - 将引理作为解答的一部分，由主 Verifier 在验证整个解答时进行处理，如果某个引理成立，就将其输出，让Orchestrator调用 ProblemMemory的类方法写入artifact
   - 如果验证失败，则分析对解答过程正确性的影响，把失败原因写入错误报告，verifier判断路由，转交给generator或reviser
   - 如果 Generator 提出了多个引理，部分成功部分失败，只把成功的存入 ProblemMemory
   - Verifier 在验证时只负责审查和问题暴露，如果Generator 在提出引理时并没有提供完整证明（只给了证明概要），则不进行补充证明、拒绝存储该引理，并将该问题在report中提出
   - Verifier 在验证整个解答时，同时处理 Generator 提出的 `<lemma>`。验证成功的引理使用 ProblemMemory的类方法写入artifact下的markdown文件；验证失败的引理，将失败原因作为“教训”放到错误报告中反馈给 Generator。Generator 提出的引理如果只给了证明概要，Verifier 不补充证明，直接拒绝并指出问题。
   - 路由逻辑是：CRITICAL_FLAW → Generator，MINOR_FLAW → Reviser。如果引理失败属于 CRITICAL（因为依赖错误的前提），应该走 Generator；如果只是证明细节不严谨但引理本身正确，可能走 Reviser。这个判断需要 Verifier 给出明确的 verdict，允许区分。
   - **引理写入**：让 Verifier 在 `<verified_lemmas>` 中完整复制 Generator 自证的 `<lemma>` 内容（包括 Proof ），而不仅仅是输出条件结论；如果Generator 在输出 `<lemma>` 时引用了artifact中的结论则注明引用路径，方便验证是否正确引用（包括引用位置和引用内容）；如果是自证的也注明 —— 要包含 `Source:` 字段
2. **子 agent 的角色与生命周期**
   - Searcher 是作为工具被 Generator 调用，作为一个独立 agent，负责检索、清洗论文，并返回 markdown 文件
   - 如果 Searcher 是独立 agent，Generator负责启动它，由 Generator 在需要时调用
   - 子 agent 的返回值（markdown 文件）的第一层暴露作为主 Agent （generator）的输入而被引用
3. **分层暴露的实现细节**
   - Agent 可以按需加载‘第二层’，由 LLM 决定使用定义的工具来实现读取。在 Prompt 中告知 Agent 如何发出加载请求，并且需要在 Agent 的调用循环中处理这种特殊输出（类似工具调用）
   - Agent 自己组装 Prompt 时去读取artifact中各引理的第一层
4. **记忆与分层暴露的耦合**
   - ProblemMemory 中存储的是“概述”，而分层暴露指的是在 Prompt 中动态注入。由 Orchestrator 来负责将“概述”统一注入到 Prompt。
5. **最终输出与引用追踪**
   - 当 Generator 调用 Searcher 并读取论文时，该论文的引用信息（arXiv ID, title, authors）应该被Searcher agent记录并存储到返回的 markdown 文件的第三层。
   - 段落级引用的插入：在有引用的每个段落后用 [1] 形式标注来源。要求 Generator 在生成解答时输出当前段落的信息来自哪个文件（定位到artifact即可，最后由Orchestrator中最后的final模块将artifact路径替换为所引用文件的第三层内容，即标准的正规引用格式），最终输出的 markdown 文件需要包含一个 ## References 章节，列出所有引用。需要按出现顺序编号，需要支持 BibTeX 导出。
   - 在Generator Prompt 中强制要求：**任何引用外部知识（论文、引理）时，必须紧跟着写出 `[cite:文件路径]`**，其中文件路径是 Searcher 返回的路径或引理文件路径。

- **重复/矛盾标注**：
  - **关于 Searcher 的去重责任**：阶段2“进一步明确”中多次提到 Searcher 负责去重，一致。无矛盾。
  - **关于 Verifier 对引理的处理**：阶段1说“Verifier 优先验证引理”，阶段3说“Verifier 在验证整个解答时，同时处理 Generator 提出的 `<lemma>`”，一致，无矛盾。

## 3. 工具/subagent层

### 3.1 状态与记忆 - 扩展状态模型（ProofState → ProblemMemory）

**你已明确的：**
- Generator、Reviser、Verifier 全员均可调用工具（查阅文献、写代码试错）

**进一步明确：**
2. **“分层暴露”的具体粒度**
   - 读取工具的参数只接收文件路径。如果 Agent 想要读取某个引理的“完整证明”（第二层），工具应该只返回第二层部分。其他层的读取也需要实现类似的工具。

### 3.2 执行层 - 将三个节点函数重构为有状态的 Agent 对象

**进一步明确：**
3. **Agent 的初始化与依赖注入**
   - 当 Generator/Reviser 获得工具调用能力后，需要像 Verifier 那样支持**多轮工具调用**（即一次 Agent 调用中可以多次使用工具）
   - 如果 Generator/Reviser 未来需要工具调用，工具列表应该配置（各个agent的工具/subagent列表不同），但同时需要引入硬性的动作截断机制，限制工具调用次数（例如每轮最多调用 5 次）。
   - 需要为每个 Agent 设置独立的 max_tool_rounds。
   - search-agent负责去重（避免重复检索同一篇论文）
   - Searcher 作为独立 subAgent，由 Generator 通过工具调用启动。Searcher 负责检索、清洗论文，返回 markdown 文件，并负责去重。Searcher 产出的结果写入 artifact。
   - Searcher 负责去重，避免重复检索同一篇论文。将当前问题的 ProblemMemory 维护的内容作为输入告诉Searcher agent，这样 Searcher 可以检查 ProblemMemory.papers 目录下是否已经存在该论文。
   - Searcher 返回的 markdown 文件应该遵循相同的三层结构：
     - 第一层 (YAML frontmatter)：定理/引理的使用条件和结论，简洁明了
     - 第二层：提取的定理/引理完整内容。
     - 第三层：引用的原文完整证明过程内容，arXiv ID, title, authors

### 3.3 状态与记忆 - 实现问题级记忆的读写与分层暴露

**你已明确的：**
- 利用子 agent（如 Searcher）检索、筛选、提取文献，返回 markdown 文件，开头包含简要概述，实现分层暴露（类似“Skill”的三层按需加载）。

**进一步明确：**
2. **子 agent 的角色与生命周期**
   - Searcher 是作为工具被 Generator 调用，作为一个独立 agent，负责检索、清洗论文，并返回 markdown 文件
   - 如果 Searcher 是独立 agent，Generator负责启动它，由 Generator 在需要时调用
   - 子 agent 的返回值（markdown 文件）的第一层暴露作为主 Agent （generator）的输入而被引用
3. **分层暴露的实现细节**
   - Agent 可以按需加载‘第二层’，由 LLM 决定使用定义的工具来实现读取。在 Prompt 中告知 Agent 如何发出加载请求，并且需要在 Agent 的调用循环中处理这种特殊输出（类似工具调用）
5. **最终输出与引用追踪**
   - 当 Generator 调用 Searcher 并读取论文时，该论文的引用信息（arXiv ID, title, authors）应该被Searcher agent记录并存储到返回的 markdown 文件的第三层。