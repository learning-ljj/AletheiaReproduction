"""编排器：Generator / Verifier / Reviser 调用封装与主循环。"""

from typing import Callable

from src.core.state import ProofState, VerificationDecision, VerificationLog
from src.models.llm_client import LLMClient, LLMResponse, create_llm_client
from src.tools.registry import execute_tool, get_tool_schemas
from src.utils.logger import append_log, print_log_entry
from src.utils.parser import (
    extract_bug_report,
    extract_preliminary_solution,
    parse_verification_verdict,
)


def call_generator(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    lesson: str | None = None,
) -> LLMResponse:
    """调用 Generator 生成初始（或重试）解答。

    Args:
        llm_client: LLM 客户端。
        prompts: 加载的 prompts.yaml 字典。
        problem_text: 原始数学问题。
        lesson: 若为 CRITICAL_FLAW 重试，传入上一轮的 bug_report 作为教训。

    Returns:
        LLMResponse（含 reasoning_content 思维链和 content 最终解答）。

    Note on lesson injection:
        lesson 直接附加在 user 消息末尾，而非作为第二条 user 消息。
        原因：DeepSeek / OpenAI API 要求 messages 角色严格交替（user/assistant）；
        连续两条 user 消息是非法格式，可能导致 API 报错或第二条被静默忽略。
    """
    if lesson:
        user_content = (
            problem_text
            + "\n\n---\n**Note: Your previous attempt was rejected due to critical flaws. "
            "Avoid repeating the following mistakes in your new attempt:**\n\n"
            + lesson
        )
    else:
        user_content = problem_text

    messages: list[dict] = [
        {"role": "system", "content": prompts["generator"]["system"]},
        {"role": "user", "content": user_content},
    ]
    return llm_client.chat(messages, thinking=True)


def call_verifier(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    proof_text: str,
    reasoning_content: str,
    tool_schemas: list[dict],
    tool_executor: Callable[[str, dict], str],
) -> tuple[str, VerificationDecision, str, list[dict], str]:
    """调用 Verifier 验证解答（三阶段：初读→工具调用→汇总）。

    Phase 1 (单轮 thinking)：初步分析，识别需工具验证的任务并列出清单。
    Phase 2 (thinking + 多轮工具调用)：对清单中的任务逐一验证。
    Phase 3 (单轮 thinking)：汇总所有发现，输出最终裁决。

    Returns:
        (full_verification_text, decision, bug_report, tool_calls_trace, phase1_analysis)
    """
    solution_body = extract_preliminary_solution(proof_text)
    phase1_content = prompts["verifier"]["phase1_user"].format(
        problem_statement=problem_text,
        reasoning_content=reasoning_content,
        solution=solution_body,
    )

    messages: list = [
        {"role": "system", "content": prompts["verifier"]["system"]},
        {"role": "user", "content": phase1_content},
    ]

    # --- Phase 1: 初读分析，无工具 ---
    print("  [Verifier Phase 1] Initial analysis (thinking only)...", flush=True)
    phase1_resp = llm_client.chat(messages, thinking=True)
    messages.append({
        "role": "assistant",
        "content": phase1_resp.content or None,
        "reasoning_content": phase1_resp.reasoning_content or None,
    })

    # --- Phase 2: 多轮工具调用验证 ---
    messages.append({"role": "user", "content": prompts["verifier"]["phase2_user"]})
    print("  [Verifier Phase 2] Tool verification (multi-turn)...", flush=True)
    phase2_resp = llm_client.chat_with_tools(messages, tool_schemas, tool_executor)
    # chat_with_tools 已将所有工具轮次追加到 messages 中

    # 清除 reasoning_content，避免新 turn 带宽浪费（DeepSeek API 要求）
    llm_client.clear_reasoning_content(messages)

    # --- Phase 3: 汇总裁决，无工具 ---
    messages.append({"role": "user", "content": prompts["verifier"]["phase3_user"]})
    print("  [Verifier Phase 3] Consolidating verdict...", flush=True)
    phase3_resp = llm_client.chat(messages, thinking=True)

    full_text = phase3_resp.content or ""
    # Bug #3 修复：Phase 3 的 content 有时为空（LLM 将输出全放在 reasoning_content 中）。
    # 在直接默认 MINOR_FLAW 之前，最多重试 2 次；若仍为空则 fallback 到 reasoning_content。
    _MAX_PHASE3_RETRIES = 2
    if not full_text.strip():
        for _attempt in range(_MAX_PHASE3_RETRIES):
            print(
                f"  [Verifier Phase 3] Content empty (attempt {_attempt + 1}/"
                f"{_MAX_PHASE3_RETRIES}), retrying...",
                flush=True,
            )
            phase3_resp = llm_client.chat(messages, thinking=True)
            full_text = phase3_resp.content or ""
            if full_text.strip():
                break
        else:
            # 全部重试失败：使用 reasoning_content 作为最后手段
            fallback = phase3_resp.reasoning_content or ""
            if fallback.strip():
                print(
                    "  [Verifier Phase 3] Using reasoning_content as fallback "
                    "(content empty after all retries).",
                    flush=True,
                )
                full_text = fallback
    decision = parse_verification_verdict(full_text)
    bug_report = "" if decision == VerificationDecision.CORRECT else extract_bug_report(full_text)
    phase1_analysis = phase1_resp.content or ""

    return full_text, decision, bug_report, phase2_resp.tool_calls_trace, phase1_analysis


def call_reviser(
    llm_client: LLMClient,
    prompts: dict,
    problem_text: str,
    previous_solution: str,
    bug_report: str,
) -> LLMResponse:
    """调用 Reviser 根据 Verifier 反馈修正解答。

    messages 模拟多轮对话：system → user（问题）→ assistant（旧解答）→ user（修正指令）。

    Returns:
        LLMResponse（含 reasoning_content 和 content）。
    """
    messages: list[dict] = [
        {"role": "system", "content": prompts["reviser"]["system"]},
        {"role": "user", "content": problem_text},
        {"role": "assistant", "content": previous_solution},
        {
            "role": "user",
            "content": prompts["reviser"]["correction_instruction"] + "\n\n" + bug_report,
        },
    ]
    return llm_client.chat(messages, thinking=True)


class AletheiaAgent:
    """Aletheia 迭代精炼 Agent：Generator → Verifier → (Reviser | Generator) 循环。"""

    def __init__(self, config: dict, prompts: dict, stream_file=None):
        self.llm_client = create_llm_client(config, stream_file=stream_file)
        self.prompts = prompts
        self.max_turns: int = config.get("agent", {}).get("max_turns", 5)
        self.tool_schemas = get_tool_schemas()
        self.tool_executor = execute_tool

    @staticmethod
    def _record(state: ProofState, **kwargs) -> None:
        """创建 VerificationLog 并追加到 state 历史 + JSONL 日志 + 控制台格式化输出。"""
        entry = VerificationLog(**kwargs)
        state.history.append(entry)
        append_log(state.problem_id, entry)
        print_log_entry(entry)

    def solve(self, problem_id: str, problem_text: str) -> ProofState:
        """执行完整的迭代精炼循环，返回最终 ProofState。"""
        state = ProofState(problem_id=problem_id, problem_text=problem_text)
        _W = 72
        print(f"\n{'═' * _W}")
        print(f"  ALETHEIA AGENT  |  Problem: {problem_id}  |  Max Turns: {self.max_turns}")
        print(f"{'═' * _W}")

        # 1. Generator 初始生成
        print("\n>>> [INIT] Calling GENERATOR...", flush=True)
        response = call_generator(self.llm_client, self.prompts, problem_text)
        state.current_proof = response.content
        self._record(state, turn_id=0, agent_node="GENERATOR",
                     content=response.content, extracted_cot=response.reasoning_content)

        for turn in range(self.max_turns):
            state.iteration_count = turn + 1

            # 2. Verifier 验证（三阶段）
            print(f"\n>>> [Turn {turn + 1}/{self.max_turns}] Calling VERIFIER...", flush=True)
            last_reasoning = state.history[-1].extracted_cot or ""
            verification_text, decision, bug_report, tool_trace, phase1_analysis = call_verifier(
                self.llm_client, self.prompts, problem_text,
                state.current_proof, last_reasoning,
                self.tool_schemas, self.tool_executor,
            )
            self._record(state, turn_id=turn + 1, agent_node="VERIFIER",
                         full_verification_text=verification_text,
                         phase1_analysis=phase1_analysis,
                         decision=decision, bug_report=bug_report,
                         tool_calls_trace=tool_trace)

            # 3. 状态机路由
            if decision == VerificationDecision.CORRECT:
                print(f"\n>>> ✅ [CORRECT] Solution accepted after {turn + 1} turn(s).")
                state.final_answer = state.current_proof
                break

            elif decision == VerificationDecision.MINOR_FLAW:
                print(f"\n>>> ⚠️  [MINOR_FLAW] Routing to REVISER...", flush=True)
                response = call_reviser(
                    self.llm_client, self.prompts, problem_text,
                    state.current_proof, bug_report,
                )
                state.current_proof = response.content
                self._record(state, turn_id=turn + 1, agent_node="REVISER",
                             content=response.content, extracted_cot=response.reasoning_content)

            else:  # CRITICAL_FLAW
                print(f"\n>>> ❌ [CRITICAL_FLAW] Routing to GENERATOR (with lesson)...", flush=True)
                response = call_generator(
                    self.llm_client, self.prompts, problem_text,
                    lesson=bug_report,
                )
                state.current_proof = response.content
                self._record(state, turn_id=turn + 1, agent_node="GENERATOR",
                             content=response.content, extracted_cot=response.reasoning_content)

        if state.final_answer is None:
            print(f"\n>>> ⏹  Max turns ({self.max_turns}) exhausted. No CORRECT verdict reached.")
        return state
