"""AletheiaAgent 门面：负责装配依赖并委托 Orchestrator。"""

from src.core.finalizer import build_final_output
from src.core.orchestrator import Orchestrator
from src.core.pipeline import call_final_assessor, call_generator, call_reviser, call_verifier
from src.core.state import ProofState
from src.models.llm_client import _UNSET as _STREAM_UNSET
from src.models.llm_client import create_llm_client
from src.tools.registry import execute_tool, get_tool_schemas
from src.utils.logger import append_raw_event


class _PipelineAdapter:
    """把函数式 pipeline 封装为对象接口，便于注入 Orchestrator。"""

    def __init__(self, llm_client, prompts, tool_schemas, tool_executor):
        self.llm_client = llm_client
        self.prompts = prompts
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor

    def call_generator(self, problem_text: str, lesson: str | None = None):
        # 这里把固定依赖绑定到实例，避免 Orchestrator 知道底层 LLM 细节。
        return call_generator(self.llm_client, self.prompts, problem_text, lesson=lesson)

    def call_verifier(self, problem_text: str, proof_text: str):
        # Verifier 只接收题目与解答正文，不再传入 reasoning_content。
        return call_verifier(
            self.llm_client,
            self.prompts,
            problem_text,
            proof_text,
            self.tool_schemas,
            self.tool_executor,
        )

    def call_reviser(self, problem_text: str, previous_solution: str, verification_report: str):
        return call_reviser(
            self.llm_client,
            self.prompts,
            problem_text,
            previous_solution,
            verification_report,
        )

    def call_final_assessor(
        self,
        problem_text: str,
        current_solution: str,
        last_verifier_decision: str,
        last_verification_report: str,
    ):
        return call_final_assessor(
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
    ) -> str:
        return build_final_output(
            success=success,
            solution_text=solution_text,
            failure_reason=failure_reason,
            partial=partial,
            assessment_output=assessment_output,
        )


class AletheiaAgent:
    """Aletheia 高层门面，内部委托 Orchestrator。"""

    def __init__(self, config: dict, prompts: dict, stream_file=_STREAM_UNSET):
        # 默认不传 stream_file：沿用 LLMClient 的默认行为（stdout 实时流式输出）。
        # 仅当调用方显式传入 None 时，才禁用流式输出。
        self.llm_client = create_llm_client(config, stream_file=stream_file)
        self.prompts = prompts
        self.max_turns: int = config.get("agent", {}).get("max_turns", 5)
        self.tool_schemas = get_tool_schemas()
        self.tool_executor = execute_tool

        # 在构造阶段完成依赖装配：solve 只负责创建状态并委托运行。
        pipeline_adapter = _PipelineAdapter(
            self.llm_client,
            self.prompts,
            self.tool_schemas,
            self.tool_executor,
        )
        self.orchestrator = Orchestrator(
            max_turns=self.max_turns,
            pipeline=pipeline_adapter,
            logger=_LoggerAdapter(),
            finalizer=_FinalizerAdapter(),
        )

    def solve(self, problem_id: str, problem_text: str, ground_truth: str | None = None) -> ProofState:
        """创建状态并委托 Orchestrator 执行。"""
        state = ProofState(problem_id=problem_id, problem_text=problem_text, ground_truth=ground_truth)
        return self.orchestrator.run(state)
