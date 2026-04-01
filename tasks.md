# ResearchMathAgent 重构任务分解计划 (MVP 落地路线)

这份任务清单旨在将 `architecture.md` 的概念与**现存代码库**进行深度融合。请执行代理 (Engineering LLM) 每次只领取**一个**任务。在执行每个任务时，必须**严格优先阅读**“现存代码剖析”中指定的旧文件，在理清现有逻辑后，再进行修改或创建。

---

## 阶段 1：状态管理与全局上下文 (State & Context)

### Task 1: 基础状态数据模型重构 (`states/state.py`)
- **现存代码剖析 (须优先阅读)**：阅读 `src/core/state.py`。原有的 `ProofState` 包含 `history: list[VerificationLog]` 这种长文本列表，导致对象臃肿，无法序列化为轻量快照。
- **负责的具体功能**：将状态流转的标量数据与其对应的历史记录剥离。定义轻量化、带有类型校验的数据类，以记录当前推演的轮数、状态机制等，防止后续流转发生字段变异。
- **类名/函数名**：
  - `class RunStatus(str, Enum)`
  - `class VerificationDecision(str, Enum)`
  - `class ProofState(BaseModel)`
- **输入参数与输出结果格式**：
  - 输入：无直接输入，只定义模型 Schema。包含字段 `problem_id: str`, `iteration_count: int`, `status: RunStatus`, `current_proof_path: str`, `last_verifier_decision: VerificationDecision | None` 等。
  - 输出格式：通过 `.model_dump_json()` 输出符合 Schema 的 JSON 字符串。
- **测试设计**：
  - **冒烟测试**：使用合法字段实例化 `ProofState`，并且无报错抛出。
  - **功能测试 (真实场景模拟)**：模拟解题过程中（例如刚执行完第一轮 Verifier 判别为 MINOR_FLAW），尝试实例化 `ProofState`，其中故意将 `iteration_count` 设置为非整型字符串 `"第一轮"`，预期输出是 Pydantic 抛出 `ValidationError`；修正后重新实例化并调用 `model_dump_json()`，预期输出一个严格的 JSON 字符串以备落盘。
- **核心知识点标注**：`pydantic.BaseModel` - 用于运行时自动数据类型校验，避免字典形式在链路传递时产生的“键/值漂移”。

### Task 2: 题目级记忆中枢实现 (`states/problem_memory.py`)
- **现存代码剖析 (须优先阅读)**：阅读 `src/utils/logger.py`（里面零散的追加写 jsonl）。旧系统缺乏对单题多模态产物（引理、文献、错误报告）的统一统筹。
- **负责的具体功能**：作为单道题目的数据管家（通过上下文变量挂载），提供标准化的文件夹初始化、状态保存、以及三级目录产物（lemma/paper/errors）的原子写入接口和读取 Layer 1 摘要的索引接口。
- **类名/函数名**： `class ProblemMemory`
  - `init_dirs(self) -> None`
  - `save_state(self, state_json: str) -> None`
  - `load_state(self) -> dict`
  - `append_history(self, event: dict) -> None`
  - `add_lemma(self, lemma_markdown: str) -> str`
  - `add_paper(self, paper_markdown: str, file_name: str) -> str`
  - `list_layer1_summaries(self, kind: str) -> list[dict]`
- **输入参数与输出结果格式**：
  - 输入：`__init__(problem_id: str, root_dir: str="runs")`。各类 `add_` 方法输入结构化字符串。
  - 输出格式：`add_` 类方法返回相对磁盘路径（如 `runs/{id}/artifact/lemmas/001.md`），`list_layer1_summaries` 返回包含 frontmatter 解析内容的列表 `list[dict]`。
- **测试设计**：
  - **冒烟测试**：实例化 `ProblemMemory("test_prob_01")`，调用 `init_dirs` 并检查文件系统是否存在对应的文件夹。
  - **功能测试 (真实场景模拟)**：使用 `pytest` 和 `tmp_path` 模拟在解题时 Generator 提交了一个引理。调用 `add_lemma` 写入包含标准 Yaml 前缀的 markdown。预期输出是硬盘上生成指定文件，且调用 `list_layer1_summaries("lemmas")` 能够成功读取出该引理的条件与结论（YAML 提取）。
- **核心知识点标注**：`contextvars.ContextVar` - 用于实现避免层层传参的线程级“当前题目沙盒变量”；`pathlib.Path` - 用于跨平台安全的系统目录原子读写。

### Task 3: 阶段 1 集成测试 (状态与存储合并)
- **负责的具体功能**：确保 `ProofState` 实例与 `ProblemMemory` 在一次完整的问题流转周期内运作正常，没有存偏或丢失数据。
- **类名/函数名**：`test_phase1_state_management()`
- **输入参数与输出结果格式**：不适用特定的输入参数，利用测试框架运行。输出即断言无任何 Traceback 或字段错误。
- **测试设计**：
  - **阶段集成测试**：在测试用例中：
    1. 给出一道虚拟 IMO 题目 ID，新建 `ProblemMemory`，挂载进 `ContextVar`。
    2. 创建 `ProofState` 描述刚完成第一轮 Generator 的状态。
    3. 序列化 `ProofState` 并交由 `ProblemMemory` 存入 `state.json`。
    4. 追加一条 `{"action": "generate_start"}` 传入 `history.jsonl`。
    5. 从硬盘读取加载复原验证数据一致性！预期结果是不抛异常，断言各个字段绝对相等。

---

## 阶段 2：解析层增强与分层文件读写 (Parsers & IO)

### Task 4: 多重 XML 标签正则解析器 (`utils/parsing/parser.py`)
- **现存代码剖析 (须优先阅读)**：阅读 `src/utils/parser.py`。目前的 `text.find` 技术只能切取首个标签包裹的内容。假如模型在一段文字输出了两个 `<lemma>`，第二个将被抛弃。
- **负责的具体功能**：通过纯正则匹配引擎，全量提取同级、多重复数出现的特定 XML 标签，确保没有任何 Generator 自证的引理或结论被遗漏。
- **类名/函数名**：
  - `extract_xml_tags(text: str, tag: str) -> list[str]`
  - `parse_lemmas_from_solution(solution_text: str) -> list[str]`
  - `parse_verified_lemmas(text: str) -> list[dict]` 
- **输入参数与输出结果格式**：
  - 输入：模型长篇返回的非结构化包含 XML 的 `text: str`。
  - 输出格式：被解包好的 `list[str]` 或对于较复杂的标签提取组合后输出 `list[dict]` 结构。
- **测试设计**：
  - **冒烟测试**：传入只有一个 `<tag>hi</tag>` 的文本，预期返回 `["hi"]`。
  - **功能测试 (真实场景模拟)**：模拟 Generator 输出：“这里我给出两个引理：<lemma>证明1使用n=1...</lemma>然后还有<lemma>\n推论2中...\n</lemma>。最后得到方案...”。通过函数提取预期输出为长度为 `2` 的数组，且换行符必须被完好包裹其中没有截断失真。
- **核心知识点标注**：`re.finditer` 结合 `re.DOTALL` - 允许非贪婪的多轮正则捕获，并允许 `.` 匹配跨越多行的换行符。

### Task 5: Markdown 分层读写引擎 (`utils/parsing/markdown_layer.py`)
- **现存代码剖析 (须优先阅读)**：当前项目中不存在。需结合 `architecture.md` 中对于 `lemma` 和 `paper` 的三层结构（YAML Frontmatter + Layer2 正文 + Layer3 来源元信息）设计。
- **负责的具体功能**：保证后续沉淀的引理和文献均可被程序化结构切片，只存留“摘要”在主索引内存（Layer 1），需要深入看证明时由工具函数从该系统读取正文（Layer 2）。
- **类名/函数名**：
  - `parse_markdown_layers(content: str) -> dict` 
  - `build_markdown_layers(frontmatter: dict, layer2: str, layer3: str) -> str`
- **输入参数与输出结果格式**：
  - 输入：解析时输入标准三段式合体的 Markdown `str`。构筑时输入前中后分离信息数据。
  - 输出格式：前者返回包含 `{"frontmatter": dict, "layer2": str, "layer3": str}` 的字典。后者返回物理文件该有的 Markdown 文本格式 `str`。
- **测试设计**：
  - **冒烟测试**：传入一个含有 `---` 包含一行 YAML 的极简 markdown 进行提取反解测试。
  - **功能测试 (真实场景模拟)**：注入一篇具有多级标题和 LaTeX 宏包正文的长难文献文本（其中包含其它 Markdown Heading 锚点，但核心需要通过正则和指定 Header 的组合切除锚定）。验证解析引擎是否正确剥离了不含 `---` 的 YAML 纯字典、完整正文以及尾随 Reference，预期输出为 3 个 Key 对应的字段不空且没有错位粘连。
- **核心知识点标注**：`yaml.safe_load` - 健壮解析 YAML 以提取 Metadata；灵活运用字符串基于标志性 Headers 锚点进行安全 Split 提取。

### Task 6: 桥接分层读取机制 (`tools/artifact_reader.py` 与 `registry.py`)
- **现存代码剖析 (须优先阅读)**：阅读旧版 `src/tools/registry.py`，查看 OpenAPI Function Schema 之前的定义方式及入参处理。
- **负责的具体功能**：提供开放给 Agent 使用读取按需暴露的武库。必须确保精准读取，只出该出的那一部分内容来阻断大文献导致的 Token 灾难。
- **类名/函数名**：
  - `read_artifact_layer1(path: str) -> str`
  - `read_artifact_layer2(path: str) -> str`
  - 添加工具的 Schema 声明字典，修改 `registry.py` (拆分为局部载入或全局声明皆可)。
- **输入参数与输出结果格式**：
  - 输入：模型推演得到的所需文献库的绝对/相对路径名 `path: str`。
  - 输出格式：所请求层级的纯内容 `str`。倘若是不存在的文件或存在非法的目录跳跃（如 `../`），务必返回包裹成人类自然语言的报错提示符 `str`，而勿引发代码执行终端 Crash 崩溃。
- **测试设计**：
  - **冒烟测试**：给出一个利用 Task 5 引擎生成的固定盘文件，看读取函数能否返回其正文文本。
  - **功能测试 (真实场景模拟)**：模拟大模型产生幻觉/意图投毒情况，传入了一个莫须有路径 `runs/not_exist_99/lemmas/99.md` 以及具有穿透威胁的 `../../../../etc/passwd`，程序必须将其拦截化解并预期返回一段如 `Error: Path not found or illegal access`，以便把这个挫败信息抛回给 Agent 在下一轮进行自行思考纠正。
- **核心知识点标注**：`Path.resolve()` 安全校验过滤体系 - 抵御潜在的路径穿越安全越权访问行为。

### Task 7: 阶段 2 集成测试 (解析读写流环绕联调)
- **负责的具体功能**：打通 Parser -> LayerEngine -> ArtifactReader -> ProblemMemory 机制。
- **测试设计**：
  - **阶段集成测试**：编写包含这四者的联合单元测试脚本。生成一段含假数据的巨大 `LLM` 日志体 -> 利用 Parser 剥离出数个 `<lemma>` 文本 -> 拆装并利用 LayerEngine 结合成符合要求的 Markdown 文本 -> 将其通过 Problem Memory 保存进临时 `tmp_path` 模拟的 Artifacts 目录下 -> 再使用 `read_artifact_layer2` 根据取得的路径从磁盘反向精准捞出未带头尾标识的纯核心正文。预期过程中无一脱节错录的抛出异常并保证流转前后的字符串还原程度一致。

---

## 阶段 3：智能体基础与子智能体网络 (Agents Framework & Searcher)

### Task 8: 智能体状态基类抽象封装 (`agents/base.py`)
- **现存代码剖析 (须优先阅读)**：告别 `src/core/pipeline.py` 此等以僵硬函数传导参数流控制大模型的范式；仔细阅览原 `src/models/llm_client.py` 提供的方法接口：尤其是 `chat_with_tools`。
- **负责的具体功能**：搭建带有 ReAct 思想核心的内卷长存的 Agent 地基。内部负责将单题的一轮解答任务维护在一个具有生命周期的 `messages` 短期私有记忆沙盒中。直到由于没 Tool 被触发或触发总数逼近阀值后向外部吐放最后定案。
- **类名/函数名**：`class BaseAgent`
  - `__init__(self, llm_client, tools: list[dict], max_tool_rounds: int)`
  - `reset_stage_memory(self) -> None`
  - `run(self, payload: dict) -> str` (负责内部不断向 LLMClient 发送 `self.messages`)
- **输入参数与输出结果格式**：
  - 输入：构建所需的所有外部 `tools` 和包含上下文 `payload` 提示词。
  - 输出格式：当判断不含 Tool_calls 或超限后，将最后一手的最终文本回答以 `str` 抛返。
- **测试设计**：
  - **冒烟测试**：挂载一个只回答一段毫无 Tool 信息文字的 Fake/Mock LLM Client 走通其 run 方法是否能即刻跳出。
  - **功能测试 (真实场景模拟)**：注入包含两个虚化数学加算法 `tools` 的子类。在执行前注入一个可被利用的 Mock Client （能够依据 Tool 定义自动传回两次中间调用意向），输入提示“解答之前请用计算器加2之后加3”。预期测试输出：检查运行完毕后通过探查 `self.messages` 记录，其中务必精确保留了其按顺序经历 user-prompt -> tool-call -> tool-return 的串行短留痕链路，及调用 `reset_stage_memory` 后记录的强制清零归寂能力。
- **核心知识点标注**：While 大循环控制与 `tool_call_id` 拼装 - 面向对象化私隐状态控制的基础实现法门。

### Task 9: 文献自动打捞小队 (`agents/searcher.py`)
- **现存代码剖析 (须优先阅读)**：阅读原来暴力抓取截断且没状态感知的 `src/tools/web_search.py` 中 `read_arxiv_latex` 等组件。
- **负责的具体功能**：演变升格成为独立思考工作网络的一员（由于并非主工作主轴，它不会直接被架构抛送问题）。只需执行针对目标意图资料的寻找->理解提炼要点->转为 3 层构架的 Markdown 模型落盘的操作。并承揽相同资料重复提取的防火防重灾设计任务。
- **类名/函数名**：`class SearcherAgent(BaseAgent)`
  - `execute_search_task(self, query: str) -> str` (继承调用内部的 run 或单独定制)
- **输入参数与输出结果格式**：
  - 输入：通过上一级父 Agent 要求代查传下来的含有意图的自然语言 `query: str`。
  - 输出格式：为控制父级 Token 极简通告语，形如："已成功检索此定理文献，摘要与出处存储落盘于 relative_path/XXX.md 中待用"。
- **测试设计**：
  - **冒烟测试**：传入一串普适化的需求要求搜索某概念证实是否平稳不报错进行。
  - **功能测试 (真实场景模拟)**：先人为给 ContextVar 里的 `ProblemMemory` 垫入一个具备某 `arxiv_id` 的虚拟 Layer1 报告数据代表已曾经调用拿过此文件；接着唤起此实例命令它按相关词查。预期输出：由于 Idempotency 去重设计的判断保护，它须在前置检查短路阻绝任何实外网 API 开销的抛出！仅仅回执已存在的旧沙盒路径宣告完毕。
- **核心知识点标注**：Idempotency 防重截留设计 - 避免对于类似论文和定理因父级陷入“死钻牛角尖”无限 Tool 工具空耗 Token 死环。

### Task 10: 构建跨端桥接调度器 (`tools/searcher_bridge.py`)
- **现存代码剖析 (须优先阅读)**：察看当前的 Tool 工具向核心注接流程。
- **负责的具体功能**：因 Generator 这个大节点无法在自己私设的逻辑里动态的编写代码 instantiate 一个旁边的 Searcher 兄弟类进行通讯调用，因此要在体系内造个供它是唤的接口将其封装为一条规范的 OpenAI Tool Scheme 管道函数。
- **类名/函数名**：
  - `call_searcher_subagent(query: str) -> str`
  - 导出并完善注册一个关于它的 JSON `searcher_bridge_schema` 参数大类给 Generator 食用。
- **输入参数与输出结果格式**：
  - 输入参数：被大模型认定的必须搜集目标描述串 `query`。
  - 输出结果格式：执行闭环所换的返回通畅路径或提示短字符串。
- **测试设计**：
  - **冒烟测试**：不引入复杂机制地单独调此方法观测是不是确实转入了实例方法。
  - **功能测试 (真实场景模拟)**：构建不连通 LLM 网络、唯作函数流转测试的小场景中注入，查验其在获取下文的接驳层畅通无卡顿死锁危险发生。
- **核心知识点标注**：Delegation 代理传导设计模式 - 将耗时或重量级高级 Agent 实质用作从属功能的隔离策略。

### Task 11: 阶段 3 集成测试 (主从工具大循环测试)
- **负责的具体功能**：联调验证一条跨级代理传递。BaseAgent-> Bridge Tool -> Searcher -> Memory 落回的传接通路。
- **测试设计**：
  - **阶段集成测试**：编写一主测试入口赋予 `BaseAgent` 假身一携带了 `searcher_bridge` 新方案的大脑；对测试替身命令："去研究一份新证明法材料再告诉我什么概念。"；断言它顺利感知应该用此新 Tool 并将内容下穿交给了底下的人！当模拟的搜索进程结束后；主系统再次从沙盒中查找到确确确实是小弟帮其生成的那份成果时；一切即验证合格并顺利。

---

## 阶段 4：算力节点重构与交接 (Core Workers)

### Task 12: 出题与引理规划者重塑 (`agents/generator.py` 与 `prompts/generator.yaml`)
- **现存代码剖析 (须优先阅读)**：翻越审读过去由于一单全揽全包，将所有题词混合写在单一一个大档内的 `config/prompts.yaml` 以及没有能力执行私我纠错的 `call_generator` 入口逻辑。
- **负责的具体功能**：将之前对它的无要求进行升格，令其负责吸收来自 Memory 提取出来的过往 Error Report 与之前就捞回的 Layer 1 (其它人的摘要与证明结论备查表)！并通过新规命令强制它的解答以严格限定的 `<lemma>` 开头自证。且外部引用务必插入规整好的 `[cite:相对路径]`。
- **类名/函数名**：`class GeneratorAgent(BaseAgent)`
  - 重写装配 prompt 的组装准备函数 `build_context` （或其覆写版本）。
- **输入参数与输出结果格式**：
  - 输入：`problem_text: str` 题目主料，加上附带过往血泪的 `error_report: str | None` （通过沙盒内含去直接读取提取组合）。
  - 输出格式：严格拥有 `<solution>` 、自证标签段、以及相关引用出处的连篇答卷结果 `str`。
- **测试设计**：
  - **冒烟测试**：对仅有基本题目大纲启动此新版组件能否如愿启动成功运转。
  - **功能测试 (真实场景模拟)**：在 Mock 替身的 `ProblemMemory` 单例里制造好含有 1篇既定 Paper，一条前次产生的 Verifier Error_Reprot 情境库。发动构建方法观察并记录传将要喂送到 LLM 大门口的那长串 Payload：重点关注它是不是完全地利用这些现实现有线索自动合围包裹在了最终的指示内提供利用查寻提示！
- **核心知识点标注**：Dynamic Context Injection 上下文反查与动态注入术 - 提供有凭有据解决思路所需要的环境要素前缀基础拼接术。

### Task 13: 威严苛刻检验官换代 (`agents/verifier.py` 与 `prompts/verifier.yaml`)
- **现存代码剖析 (须优先阅读)**：原作者将其设为独霸多层逻辑且硬写分支的一大堆冗沉判别机制环节（参考旧体 `call_verifier`）。现应并流纳回统一化智能架构中。
- **负责的具体功能**：将 Generator 交过来的解答卷抽丝剥茧。用独立 Regex 挑出 Lemma 段使用其特有的沙盒执行跑 Py 做数算推演。**必须执行针对 `[cite:x]` 的幻觉真实防呆拦截测定**；然后生成强类型路由断言与含有附带完整论述的验证报单。
- **类名/函数名**：`class VerifierAgent(BaseAgent)`
- **输入参数与输出结果格式**：
  - 输入参数：接收前方全流程的完整输出原文本解法。
  - 输出格式：将内容按规定的 `<verdict>MINOR_FLAW / CORRECT...`，有可复用就提炼入内的 `<verified_lemmas>` 契约返回出解析用的大篇幅长 `str` 回应流。
- **测试设计**：
  - **冒烟测试**：赋予一正确解答且无任何引理引用的情形下平稳下达判决流逝。
  - **功能测试 (真实场景模拟)**：丢进含一个故意缺少一半步骤瞎扯出来的伪理论的 `lemma` 数据同时塞入一被捏造出的幻觉 `[cite:a_fake_path.md]` 不存在路径地址的虚空出处证明；预期拦截机制必定在其执行判决流与 Python 证实途中将其归类列回 `MINOR_FLAW / CRITICAL_FLAW` 类别且一五一十写出是哪处产生了谬误或不可信以备报告产出！
- **核心知识点标注**：防御式检验法则与枚举状态路由分发。

### Task 14: 漏洞填补修补匠 (`agents/reviser.py` 与 `prompts/reviser.yaml`)
- **现存代码剖析 (须优先阅读)**：沿袭 `call_reviser` 作派，但弃暗投明利用最新这套拥有自驱多轮重检和拥有自己外围工具能力的特化体系来替代。
- **负责的具体功能**：当由于一些不足轻重仅需部分重写的时候触发，让它专务于用工具测定某独立区间和提供有底线的不破坏已有核心的重新补出替换工作职能范畴。
- **输入参数与输出结果格式**：接收 `error_report` 、 `previous_solution` 等等重新反馈修改后的最终文本形式解答长字符串。
- **测试设计**：
  - **功能测试 (真实场景模拟)**：传进一个仅有某一行计算写出 2+3=6 的瑕疵结果搭配明说的 Error。让它利用工具核算出它后是否可以以小幅度变工的形式输出正确的 `solution` 并附带新替换内容返回。

### Task 15: 阶段 4 集成测试 (算力单元节点穿引连线)
- **负责的具体功能**：仅以代码层面断言前线三大业务干将能够进行前后信息流无漏损接抛！不夹涉外界 LLM 和真实模型响应。
- **测试设计**：
  - **阶段集成测试**：编写包含手写串流程的代码单元测试，自己准备字符串 A 表示 Generator 已经生成好内容；让 Task 4 的 Parser 将其切开交付发派交给 Verifier ，验证 Verifier 理应返回出特定含有 Error 发送给 Reviser，再由 Reviser 的预组建函数观测是否能够成功全部捕装接收到没有任何字句错丢的关键数据内容。预期不产生任何由于缺少接口兼容的问题导致崩溃。

---

## 阶段 5：控制塔台集结及落幕产出规范 (Orchestrator Heart & Finalizer)

### Task 16: 专业引用解释映射生成器 (`utils/parsing/reference_builder.py` 与 `finalizer.py` 增强)
- **现存代码剖析 (须优先阅读)**：参考 `src/core/finalizer.py` 下极度软薄和简陋无为的收尾拼接 `build_final_output` 函数方法体构造。
- **负责的具体功能**：将之前满篇由于提示词限定而产出的 `关于这一点见[cite:runs/.../a.md]` 等原生态技术痕迹语句彻底转译并抹为人类自然平滑阅读的 `关于这一点见[1]`；通过物理寻址读出我们原 Layer 3 层隐藏保护许久的 Metadata 出处并做漂亮的末尾表单打印展示。
- **类名/函数名**：`finalize_output_with_references(solution: str, memory: ProblemMemory) -> str`
- **输入参数与输出结果格式**：
  - 输入参数：原包含未经处理标注信息的整个结果篇章 `solution`，提供检索根据的源文指针仓库管家。
  - 输出格式：完成格式翻译和底端拼贴 `## References` 完美打印准备可发表的长段 `str` 输出串。
- **测试设计**：
  - **冒烟测试**：丢个光秃秃没标示的文章，预期输出应该如不包含参考文献表单一样的未作影响。
  - **功能测试 (真实场景模拟)**：传给它这篇充满交杂引用复用的段落："一、参见由于[cite:artifact/papers/a.md]的限制，又依据[cite:artifact/papers/b.md]... 依然[cite:artifact/papers/a.md]" 预期：要求其能够经过映射解析去重将这俩同类的统一标化收拢并返回成标有 `[1]`, `[2]` 且第二次提及又显示回 `[1]` 的精确内容。于文后打印展示相应详细溯源。
- **核心知识点标注**：`re.sub` 通过引入 CallBack 重调进行引文登记复用计数编号管理置换。

### Task 17: 重写驱动系统的跳动脉搏 (`core/orchestrator.py`)
- **现存代码剖析 (须优先阅读)**：极其有必要地细细精读通览旧版的 `src/core/orchestrator.py` 与用来妥协挂带功能的代理封装件 `src/core/agent.py` (_PipelineAdapter) 间的所有控制耦合流法！
- **负责的具体功能**：把这所有的全盘架构组装在一起成为能够自行跑动的活物！剥下包袱代码。在此主动去拉起设置所有 Agent 并注入 `ProblemMemory` 提供系统唯一线程级环境供氧输送。依据最外层 `max_turns` 大圈轮换着安排人员下场（生成->解析->验证->写入或驳回找借口重制）。
- **类名/函数名**：`class Orchestrator:` 内的 `__init__`, 以及最为粗核心的方法 `def run(self, problem_id: str, problem_text: str)`
- **输入参数与输出结果格式**：
  - 输入参数：系统启动必备的解题题序 ID 及源主任务指令内容题文结构。
  - 输出格式：穿出最后阶段通过层层校验过五关斩六将产出的可复现完美学术成文解答字符串！
- **测试设计**：
  - **冒烟测试**：用极虚的返回瞬间给与一遍 Correct。
  - **功能测试 (真实场景模拟)**：不调 LLM；将此管线的类三巨头 Agent 做函数 Fake 化。使其走这条轨迹：第1轮 Generator 答复包含1个正确 Lemma 但整体存在 Error。-> 经过管内 Parser 和 Verifier 并给判定 `MINOR_FLAW` 还要测试它确实指挥 `add_lemma` 落进了盘。-> 被踢到 Reviser 加工。-> 转出通过 `CORRECT` 打破循环；经过最终 Finalizer 落档写存结束。全程追断它的每条历史足迹流是否被老老实实记载了进入。
- **核心知识点标注**：大主控状态切换流闭壳实现术 - 极其高度内聚调度层而不处理实活干系流分工隔离法。

### Task 18: 阶段 5 体系引擎集成联调冒烟跑图
- **负责的具体功能**：进行最逼近真章除没真实 API 往外的模拟全链路冲锋演训测试！
- **测试设计**：
  - **阶段集成测试**：采用纯断言校验不插手的方式，在本地开发根源以单元测试指令将其跑动启动；观测跑完之后对应的路径之下（即便全假充字）但是否各个规定的表单和层次的存储沙盒都已经老老实实并完完整整不爆乱码的在预先定好的地方存有应生成的痕迹？

---

## 阶段 6：端到端真实评测 (E2E Master)

### Task 19: 实网全架构验证演习打通阻滞 (The Final Run)
- **负责的具体功能**：此作为最后一个不包含撰定或添写框架组件主体的纯任务执行项（不修改主框架代码除非遇死胡同错处漏招）。仅需要挂好网络 API 使用深思维等具有超高素质的大型智眼用我们从头到脚新拼合出来的心脏实跑一场高光高燃难题演算试炼即可！执行脚本、观测一切、捕出藏在设计死点和盲角的小毛病！
- **测试设计 (包含任务运行指引)**：
  - **最终端到端 (E2E) 测试要求必须做到：**
    1. 使用原先配套存有的 `imobench` 等题类挑个不简单的做实景目标；利用环境下的启动 `main.py --problem_id xyz` 等正式指令起帆！
    2. 打开 `Debug` 吐放限制。在此试运行全期间：不论好坏工程 LLM 您必须**实时跟踪并摘抄记录大量的中间重要过站流变现象与产物情况**放入您的反馈总结呈文中（包括比如 `history.jsonl` 真如预期输出了没，子 Agent 调起工具时候 Payload 里到底是长什么样子传了没漏等）。
    3. 碰碰撞墙时刻（绝对会发生，必定例如某个意图没有让大模型听话或者导致了某些不严格字符使得正则崩裂抑或某 API 时限过长警铃长响）：必须记录留档在案并书写出具非常细腻针对此处情况缘由分析判定的**诊治破案探究说明**小结。
    4. 明确分析清楚究竟是卡在哪儿之后，立马回到之前那些已经“自以为写得很完善”的单档代码模块中（如某个 Regex，哪怕是某个指令说明中的某个用词）再作润色增重小幅度 Fix！然后反反复复在此轮回。
    5. 一路磕绊披除修整直致在运行结束后；亲眼见证 `runs/xyz` 中赫然满目琳琅排列着我们那优美的三层 Lemma 与搜研 Paper。最终生成了一篇含出处底角批注无半点差错的 Final.md 正统全篇章！完满达成并写个漂亮的胜赛长书复命。
- **核心知识点标注**：大架构落脚实践下的 Debug Triage / Root Cause Analysis 闭环机置实战处理应修与能力论证。个指令说明中的某个用词）再作润色增重小幅度 Fix！然后反反复复在此轮回。
    5. 一路磕绊披除修整直致在运行结束后；亲眼见证 `runs/xyz` 中赫然满目琳琅排列着我们那优美的三层 Lemma 与搜研 Paper。最终生成了一篇含出处底角批注无半点差错的 Final.md 正统全篇章！完满达成并写个漂亮的胜赛长书复命。
- **核心知识点标注**：大架构落脚实践下的 Debug Triage / Root Cause Analysis 闭环机置实战处理应修与能力论证。