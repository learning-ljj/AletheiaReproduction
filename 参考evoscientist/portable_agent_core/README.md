# Portable Agent Core

This folder contains a low-coupling reference implementation extracted from the
architecture of the EvoScientist repository.

Files:

- `shared_types.py`: shared DTOs and Protocol interfaces
- `agent_framework.py`: main-agent / subagent orchestration loop
- `literature_search.py`: literature-style search with provider fallback
- `content_extraction.py`: HTML / PDF extraction pipeline
- `resilience.py`: retry, guarded tool calls, and self-correction loop
- `demo_research_runtime.py`: minimal runnable example

Mapping back to the source repository:

- `EvoScientist/EvoScientist.py` -> runtime assembly and subagent registration
- `EvoScientist/subagent.yaml` -> subagent roles and tool ownership
- `EvoScientist/tools/search.py` -> search plus follow-up content fetching
- `EvoScientist/middleware/tool_error_handler.py` -> exception-as-data handling
- `EvoScientist/backends.py` -> bounded retries and safe execution ideas

Suggested migration order:

1. Migrate `shared_types.py` and `agent_framework.py`
2. Replace `StaticSearchProvider` with your real search provider
3. Plug `UniversalContentExtractor` behind your own content service
4. Wrap your framework's tool execution with helpers from `resilience.py`

生成的可迁移实现

Agent 驱动框架：
agent_framework.py
主 agent / subagent 交互示例：
demo_research_runtime.py
文献检索：
literature_search.py
内容提取：
content_extraction.py
容错与自我修正：
resilience.py
共享数据结构：
shared_types.py
它们参考了仓库里的哪些真实实现

Agent 运行时装配： EvoScientist.py
主运行流式入口： events.py
subagent 定义： subagent.yaml
检索与网页抓取： search.py
工具异常转可恢复消息： tool_error_handler.py
安全执行、超时回退： backends.py
你现在可以怎么学

先看 demo_research_runtime.py：最直观看“主 agent 如何委派、subagent 如何回传结构化结果、主 agent 如何继续下一轮”。
再看 agent_framework.py：这是最小可迁移的运行时骨架。
然后看 literature_search.py 和 content_extraction.py：分别对应“找文献”和“抽正文”。
最后看 resilience.py：把容错、自修正单独抽成工具层，方便塞进你自己的 agent 框架。
验证结果

我已运行 python -m compileall，这些文件可以编译。
我已运行 python -m migration_examples.portable_agent_core.demo_research_runtime，主链路已跑通。
当前环境禁止联网，所以提取阶段对 example.org 的真实访问失败；这不是代码结构问题，反而验证了容错路径会把错误变成结构化结果，再交还主 agent 继续收敛。