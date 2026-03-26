# 可移植代理核心（Portable Agent Core）

本目录包含从 EvoScientist 仓库抽取出的低耦合参考实现，便于在不同运行时中复用代理（agent）框架的核心思路。

主要文件：

- `shared_types.py`：共享的 DTO（数据传输对象）和协议接口定义。
- `agent_framework.py`：主 agent 与子 agent（subagent）编排循环的最小实现。
- `literature_search.py`：带后备策略的文献检索实现（可替换后端）。
- `content_extraction.py`：HTML / PDF 内容抽取流水线。
- `resilience.py`：重试、受保护的工具调用与自我修正循环工具。
- `demo_research_runtime.py`：最小可运行示例，展示如何组装并运行。

与原仓库的对应关系：

- `EvoScientist/EvoScientist.py` -> 运行时装配与 subagent 注册逻辑。
- `EvoScientist/subagent.yaml` -> subagent 的角色定义与工具归属。
- `EvoScientist/tools/search.py` -> 检索与后续内容抓取。
- `EvoScientist/middleware/tool_error_handler.py` -> 将异常转为可恢复的数据结构的中间件思路。
- `EvoScientist/backends.py` -> 有界重试与安全执行的实现思路。

建议迁移顺序：

1. 先迁移 `shared_types.py` 与 `agent_framework.py`，建立最小运行时骨架。
2. 用你的真实检索后端替换示例中的 `StaticSearchProvider`。
3. 将 `UniversalContentExtractor` 接入到你的内容服务（Downloader/Parser）后端。
4. 用 `resilience.py` 中的保护调用封装你框架中的工具执行路径。

快速索引（便于阅读实现）：

- 运行时与编排：`agent_framework.py`
- 演示运行：`demo_research_runtime.py`
- 文献检索：`literature_search.py`
- 内容抽取：`content_extraction.py`
- 容错工具：`resilience.py`
- 共享类型：`shared_types.py`

学习建议：

- 先运行并阅读 `demo_research_runtime.py`：直接观察主 agent 如何委派任务、subagent 如何返回结构化结果，以及主 agent 如何根据结果继续下一轮。
- 再阅读 `agent_framework.py`：这是最小且可迁移的运行时骨架，便于将功能嵌入现有系统。
- 随后查看 `literature_search.py` 与 `content_extraction.py`：理解检索链路与正文抽取的边界与错误处理。
- 最后研究 `resilience.py`：学习如何把容错与自我修正独立成可复用组件。

验证说明：

- 我已用 `python -m compileall` 对示例文件做过静态编译检查，文件可通过编译。
- 我已运行 `python -m migration_examples.portable_agent_core.demo_research_runtime`，主链路在无外网条件下展示了容错路径工作（真实下载会因无网失败，但失败被转为结构化错误并返回主 agent）。

如果需要，我可以把 `demo_research_runtime.py` 的运行步骤与示例输出整理为更详细的演练文档。 