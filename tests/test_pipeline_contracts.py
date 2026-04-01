from src.agents.generator import GeneratorAgent
from src.core.config import load_prompts


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = ""


class _FakeLLMForGenerator:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.chat_calls = 0

    def chat(self, messages, thinking=True, stream_prefix=None):
        idx = min(self.chat_calls, len(self.outputs) - 1)
        self.chat_calls += 1
        return _Resp(self.outputs[idx])

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=10, stream_prefix=None):
        idx = min(self.chat_calls, len(self.outputs) - 1)
        self.chat_calls += 1
        return _Resp(self.outputs[idx])


def test_prompt_generator_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["generator"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "<lemma>" in text
    assert "</lemma>" in text
    assert "[cite:" in text


def test_generator_agent_returns_structured_output() -> None:
    llm = _FakeLLMForGenerator([
        "<verdict>PARTIAL</verdict>\n<solution><lemma>L1</lemma>step</solution>",
    ])
    agent = GeneratorAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(problem_text="demo problem", error_lessons="avoid gap")
    assert "<verdict>" in resp.content
    assert "<solution>" in resp.content


def test_generator_agent_retries_once_when_contract_missing() -> None:
    llm = _FakeLLMForGenerator([
        "draft without tags",
        "<verdict>PARTIAL</verdict>\n<solution>fixed</solution>",
    ])
    agent = GeneratorAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=[],
        tool_executor=None,
    )

    resp = agent.run(problem_text="demo")
    assert "<solution>" in resp.content
    assert llm.chat_calls == 2



def test_prompt_verifier_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["verifier"]["phase3_user"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<verification>" in text
    assert "</verification>" in text
    assert "<verified_lemmas>" in text
    assert "</verified_lemmas>" in text
    assert "<citation_review>" in text
    assert "</citation_review>" in text



def test_prompt_reviser_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["reviser"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "[cite:" in text
