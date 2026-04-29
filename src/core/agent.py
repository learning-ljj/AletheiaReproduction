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
    configure_searcher_sources,
    execute_tool,
    format_tool_error,
    get_tool_schemas,
)
from src.tools.search_sources import build_default_source_handlers


_AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "generator": {"read_artifact_layer", "call_searcher"},
    "reviser": {"read_artifact_layer", "call_searcher"},
    "verifier": {"run_python", "read_artifact_layer", "call_citation_reviewer"},
}


def _filter_tool_schemas(tool_schemas: list[dict], allowed_names: set[str]) -> list[dict]:
    """按白名单筛选可见的 tool schema。"""
    filtered: list[dict] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name")
        if name in allowed_names:
            filtered.append(schema)
    return filtered


def _build_scoped_tool_executor(
    base_executor: Callable[[str, dict], str],
    allowed_names: set[str],
) -> Callable[[str, dict], str]:
    """构造仅允许白名单工具的执行器。"""

    def _executor(function_name: str, arguments: dict) -> str:
        if function_name not in allowed_names:
            allowed_list = sorted(allowed_names)
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
        return base_executor(function_name, arguments)

    return _executor


class AgentPipeline:
    """主链 Agent 运行时：直接装配 Generator/Verifier/Reviser 对象。"""

    def __init__(
        self,
        llm_client,
        prompts,
        tool_schemas,
        tool_executor,
        *,
        max_tool_rounds: int = 20,
    ):
        self.llm_client = llm_client
        self.prompts = prompts

        generator_allowed = _AGENT_TOOL_ALLOWLIST["generator"]
        reviser_allowed = _AGENT_TOOL_ALLOWLIST["reviser"]
        verifier_allowed = _AGENT_TOOL_ALLOWLIST["verifier"]

        generator_tools = _filter_tool_schemas(tool_schemas, generator_allowed)
        reviser_tools = _filter_tool_schemas(tool_schemas, reviser_allowed)
        verifier_tools = _filter_tool_schemas(tool_schemas, verifier_allowed)

        generator_executor = _build_scoped_tool_executor(tool_executor, generator_allowed)
        reviser_executor = _build_scoped_tool_executor(tool_executor, reviser_allowed)
        verifier_executor = _build_scoped_tool_executor(tool_executor, verifier_allowed)

        self.generator_agent = GeneratorAgent(
            llm_client=self.llm_client,
            system_prompt=self.prompts["generator"]["system"],
            tools=generator_tools,
            tool_executor=generator_executor,
            max_tool_rounds=max_tool_rounds,
        )
        self.verifier_agent = VerifierAgent(
            llm_client=self.llm_client,
            prompts=self.prompts,
            tools=verifier_tools,
            tool_executor=verifier_executor,
            max_tool_rounds=max_tool_rounds,
        )
        self.reviser_agent = ReviserAgent(
            llm_client=self.llm_client,
            system_prompt=self.prompts["reviser"]["system"],
            tools=reviser_tools,
            tool_executor=reviser_executor,
            max_tool_rounds=max_tool_rounds,
        )

    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        lemma_context_items: list[str] | None = None,
    ):
        # GeneratorAgent 对象执行。
        return self.generator_agent.run(
            problem_text=problem_text,
            error_lessons=lesson,
            lemma_context_items=lemma_context_items,
        )

    def call_verifier(self, problem_text: str, proof_text: str):
        # VerifierAgent 对象执行。
        return self.verifier_agent.run(
            problem_text=problem_text,
            proof_text=proof_text,
        )

    def call_reviser(
        self,
        problem_text: str,
        previous_solution: str,
        verification_report: str,
        lemma_context_items: list[str] | None = None,
    ):
        # ReviserAgent 对象执行。
        return self.reviser_agent.run(
            problem_text=problem_text,
            previous_solution=previous_solution,
            verification_report=verification_report,
            lemma_context_items=lemma_context_items,
        )


def _resolve_tool_round_limits(config: dict) -> int:
    resilience_cfg = config.get("resilience", {}) if isinstance(config, dict) else {}
    return int(resilience_cfg.get("max_tool_rounds", 20))


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

        configure_searcher_sources(build_default_source_handlers(config))
        tool_schemas = get_tool_schemas()
        rounds = _resolve_tool_round_limits(config)

        pipeline = AgentPipeline(
            self.llm_client,
            prompts,
            tool_schemas,
            execute_tool,
            max_tool_rounds=rounds,
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
