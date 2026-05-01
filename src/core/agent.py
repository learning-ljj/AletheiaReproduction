"""AletheiaAgent 门面：负责装配依赖并委托 Orchestrator。"""

from typing import Callable

from src.agents.generator import GeneratorAgent
from src.agents.reviser import ReviserAgent
from src.agents.verifier import VerifierAgent
from src.core.orchestrator import Orchestrator
from src.memory.state import ProofState
from src.models.llm_client import _UNSET as _STREAM_UNSET
from src.models.llm_client import create_llm_client
from src.tools.registry import (
    format_tool_error,
    ToolExecutor,
)
from src.tools.schemas import get_tool_schemas
from src.tools.search_papers.search_sources import build_search_source_handlers


_AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    # "generator": {"read_artifact", "call_searcher"},
    # "reviser": {"read_artifact", "call_searcher"},
    "generator": {"read_artifact"},
    "reviser": {"read_artifact"},
    "verifier": {"run_python", "read_artifact", "review_citation"},
}


def _filter_tool_schemas(tool_schemas: list[dict], allowed_names: set[str]) -> list[dict]:
    """按白名单筛选可见的 tool schema。"""
    filtered: list[dict] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name")
        if name in allowed_names:
            filtered.append(schema)
    return filtered


class ScopedToolExecutor:
    """带工具白名单的执行器，避免使用函数嵌套工厂。"""

    def __init__(self, base_executor: Callable[[str, dict], str], allowed_names: set[str]):
        self.base_executor = base_executor
        self.allowed_names = set(allowed_names)

    def __call__(self, function_name: str, arguments: dict) -> str:
        if function_name not in self.allowed_names:
            allowed_list = sorted(self.allowed_names)
            return format_tool_error(
                tool=function_name,
                error_code="TOOL_NOT_PERMITTED_IN_STAGE",
                message=(
                    f"Tool {function_name!r} is not permitted in this agent stage. "
                    f"Allowed: {allowed_list}"
                ),
                retryable=False,
                detail={"allowed": allowed_list},
            )
        return self.base_executor(function_name, arguments)


class AgentPipeline:
    """主链 Agent 运行时：直接装配 Generator/Verifier/Reviser 对象。"""

    def __init__(
        self,
        llm_client,
        prompts,
        tool_schemas,
        tool_executor,
        *,
        max_rounds: int = 20,
    ):
        self.llm_client = llm_client
        self.prompts = prompts

        generator_allowed = _AGENT_TOOL_ALLOWLIST["generator"]
        reviser_allowed = _AGENT_TOOL_ALLOWLIST["reviser"]
        verifier_allowed = _AGENT_TOOL_ALLOWLIST["verifier"]

        generator_tools = _filter_tool_schemas(tool_schemas, generator_allowed)
        reviser_tools = _filter_tool_schemas(tool_schemas, reviser_allowed)
        verifier_tools = _filter_tool_schemas(tool_schemas, verifier_allowed)

        generator_executor = ScopedToolExecutor(tool_executor, generator_allowed)
        reviser_executor = ScopedToolExecutor(tool_executor, reviser_allowed)
        verifier_executor = ScopedToolExecutor(tool_executor, verifier_allowed)

        self.generator_agent = GeneratorAgent(
            llm_client=self.llm_client,
            system_prompt=self.prompts["generator"]["system"],
            tools=generator_tools,
            tool_executor=generator_executor,
            max_rounds=max_rounds,
        )
        self.verifier_agent = VerifierAgent(
            llm_client=self.llm_client,
            prompts=self.prompts,
            tools=verifier_tools,
            tool_executor=verifier_executor,
            max_rounds=max_rounds,
        )
        self.reviser_agent = ReviserAgent(
            llm_client=self.llm_client,
            system_prompt=self.prompts["reviser"]["system"],
            tools=reviser_tools,
            tool_executor=reviser_executor,
            max_rounds=max_rounds,
        )


def _resolve_tool_round_limits(config: dict) -> int:
    resilience_cfg = config.get("resilience", {}) if isinstance(config, dict) else {}
    return int(resilience_cfg.get("max_rounds", 20))


class AletheiaAgent:
    """Aletheia 高层门面，内部委托 Orchestrator。"""

    def __init__(self, config: dict, prompts: dict, stream_file=_STREAM_UNSET):
        # 默认不传 stream_file：沿用 LLMClient 的默认行为（stdout 实时流式输出）。
        # 仅当调用方显式传入 None 时，才禁用流式输出。
        self.llm_client = create_llm_client(config, stream_file=stream_file)
        self.prompts = prompts

        agent_cfg = config.get("agent", {}) if isinstance(config, dict) else {}
        self.max_turns = int(agent_cfg.get("max_turns", 5))
        self.runs_root = str(agent_cfg.get("runs_root", "runs"))

        search_source_handlers = build_search_source_handlers(config)
        tool_schemas = get_tool_schemas()
        rounds = _resolve_tool_round_limits(config)

        tool_executor = ToolExecutor(search_source_handlers)

        pipeline = AgentPipeline(
            self.llm_client,
            prompts,
            tool_schemas,
            tool_executor,
            max_rounds=rounds,
        )
        self.orchestrator = Orchestrator(
            max_turns=self.max_turns,
            pipeline=pipeline,
            runs_root=self.runs_root,
        )

    def solve(self, problem_id: str, problem_text: str, ground_truth: str | None = None) -> ProofState:
        """创建状态并委托 Orchestrator 执行。"""
        state = ProofState(problem_id=problem_id, problem_text=problem_text, ground_truth=ground_truth)
        return self.orchestrator.run(state)
