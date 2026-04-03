"""AletheiaAgent 门面：负责装配依赖并委托 Orchestrator。"""

from typing import Callable

from src.agents.generator import GeneratorAgent
from src.agents.reviser import ReviserAgent
from src.agents.verifier import VerifierAgent
from src.core.finalizer import build_final_output, call_final
from src.core.orchestrator import Orchestrator
from src.core.state import ProofState
from src.models.llm_client import _UNSET as _STREAM_UNSET
from src.models.llm_client import create_llm_client
from src.tools.registry import (
    configure_searcher_sources,
    configure_tool_resilience,
    execute_tool,
    format_tool_error,
    get_tool_schemas,
)
from src.tools.search_sources import build_default_source_handlers
from src.utils.logging.logger import append_raw_event, save_final_output_markdown


_AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "generator": {"run_python", "call_searcher"},
    "reviser": {"run_python", "call_searcher", "read_artifact_layer"},
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


class _AgentRuntime:
    """主链 Agent 运行时：直接装配 Generator/Verifier/Reviser 对象。"""

    def __init__(
        self,
        llm_client,
        prompts,
        tool_schemas,
        tool_executor,
        *,
        generator_max_tool_rounds: int = 5,
        reviser_max_tool_rounds: int = 5,
        verifier_max_tool_rounds: int = 5,
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
            max_tool_rounds=generator_max_tool_rounds,
        )
        self.verifier_agent = VerifierAgent(
            llm_client=self.llm_client,
            prompts=self.prompts,
            tools=verifier_tools,
            tool_executor=verifier_executor,
            max_tool_rounds=verifier_max_tool_rounds,
        )
        self.reviser_agent = ReviserAgent(
            llm_client=self.llm_client,
            system_prompt=self.prompts["reviser"]["system"],
            tools=reviser_tools,
            tool_executor=reviser_executor,
            max_tool_rounds=reviser_max_tool_rounds,
        )

    def call_generator(
        self,
        problem_text: str,
        lesson: str | None = None,
        layer1_summaries: list[str] | None = None,
    ):
        # C31: 主路径改为 GeneratorAgent 对象执行。
        return self.generator_agent.run(
            problem_text=problem_text,
            error_lessons=lesson,
            layer1_summaries=layer1_summaries,
        )

    def call_verifier(self, problem_text: str, proof_text: str):
        # C32: 主路径改为 VerifierAgent 对象执行。
        return self.verifier_agent.run(
            problem_text=problem_text,
            proof_text=proof_text,
        )

    def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
        return self.reviser_agent.run(
            problem_text=problem_text,
            previous_solution=previous_solution,
            verification_report=verification_report,
        )

    def call_final(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return call_final(
            self.llm_client,
            self.prompts,
            problem_text,
            current_solution,
            last_verifier_decision,
            last_verification_report,
        )


class _LoggerAdapter:
    """为 Orchestrator 提供最小日志写接口。"""

    @staticmethod
    def append_raw_event(problem_id: str, payload: dict) -> None:
        append_raw_event(problem_id=problem_id, payload=payload)

    @staticmethod
    def save_final_output_markdown(problem_id: str, final_output: str) -> None:
        save_final_output_markdown(problem_id=problem_id, final_output=final_output)


class _FinalizerAdapter:
    """把函数式 finalizer 封装成对象，统一 Orchestrator 依赖接口。"""

    @staticmethod
    def build_final_output(
        success: bool,
        solution_text: str | None,
        failure_reason: str | None,
        *,
        partial: bool = False,
        assessment_output: str | None = None,
        preserve_xml: bool = False,
        references: list[str] | None = None,
        warning_summary: str | None = None,
    ) -> str:
        return build_final_output(
            success=success,
            solution_text=solution_text,
            failure_reason=failure_reason,
            partial=partial,
            assessment_output=assessment_output,
            preserve_xml=preserve_xml,
            references=references,
            warning_summary=warning_summary,
        )


class AletheiaAgent:
    """Aletheia 高层门面，内部委托 Orchestrator。"""

    def __init__(self, config: dict, prompts: dict, stream_file=_STREAM_UNSET):
        # 默认不传 stream_file：沿用 LLMClient 的默认行为（stdout 实时流式输出）。
        # 仅当调用方显式传入 None 时，才禁用流式输出。
        self.llm_client = create_llm_client(config, stream_file=stream_file)
        self.prompts = prompts
        self.max_turns: int = config.get("agent", {}).get("max_turns", 5)
        self.runs_root: str = config.get("agent", {}).get("runs_root", "runs")
        self.tool_schemas = get_tool_schemas()
        self.tool_executor = execute_tool

        resilience_cfg = config.get("resilience", {}) if isinstance(config, dict) else {}
        default_rounds = int(resilience_cfg.get("default_max_tool_rounds", 5))
        generator_rounds = int(resilience_cfg.get("generator_max_tool_rounds", default_rounds))
        reviser_rounds = int(resilience_cfg.get("reviser_max_tool_rounds", default_rounds))
        verifier_rounds = int(resilience_cfg.get("verifier_max_tool_rounds", default_rounds))

        configure_tool_resilience(
            max_attempts=int(resilience_cfg.get("tool_retry_max_attempts", 1)),
            backoff_seconds=float(resilience_cfg.get("tool_retry_backoff_seconds", 0.2)),
        )
        configure_searcher_sources(build_default_source_handlers(config))

        # 在构造阶段完成依赖装配：solve 只负责创建状态并委托运行。
        agent_runtime = _AgentRuntime(
            self.llm_client,
            self.prompts,
            self.tool_schemas,
            self.tool_executor,
            generator_max_tool_rounds=generator_rounds,
            reviser_max_tool_rounds=reviser_rounds,
            verifier_max_tool_rounds=verifier_rounds,
        )
        self.orchestrator = Orchestrator(
            max_turns=self.max_turns,
            pipeline=agent_runtime,
            logger=_LoggerAdapter(),
            finalizer=_FinalizerAdapter(),
            runs_root=self.runs_root,
        )

    def solve(self, problem_id: str, problem_text: str, ground_truth: str | None = None) -> ProofState:
        """创建状态并委托 Orchestrator 执行。"""
        state = ProofState(problem_id=problem_id, problem_text=problem_text, ground_truth=ground_truth)
        return self.orchestrator.run(state)
