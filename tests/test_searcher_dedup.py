from src.agents.base import BaseAgent


class _Resp:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeLLMClient:
    def __init__(self):
        self.chat_calls = 0
        self.chat_with_tools_calls = 0
        self.last_max_tool_rounds = None

    def chat(self, messages, thinking=True, stream_prefix=None):
        self.chat_calls += 1
        return _Resp(content="plain-result", reasoning_content="plain-reason")

    def chat_with_tools(self, messages, tools, tool_executor, max_tool_rounds=10, stream_prefix=None):
        self.chat_with_tools_calls += 1
        self.last_max_tool_rounds = max_tool_rounds
        tool_executor("dummy", {})
        return _Resp(content="tool-result", reasoning_content="tool-reason")



def test_base_agent_stage_memory_reset_between_runs() -> None:
    llm = _FakeLLMClient()
    agent = BaseAgent(llm_client=llm, system_prompt="sys", max_tool_rounds=5)

    first_output = agent.run("payload-one")
    assert first_output == "plain-result"
    first_snapshot = list(agent.messages)
    assert any("payload-one" in (m.get("content") or "") for m in first_snapshot)

    second_output = agent.run("payload-two")
    assert second_output == "plain-result"
    assert any("payload-two" in (m.get("content") or "") for m in agent.messages)
    assert not any("payload-one" in (m.get("content") or "") for m in agent.messages)



def test_base_agent_respects_max_tool_rounds() -> None:
    llm = _FakeLLMClient()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dummy",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent = BaseAgent(
        llm_client=llm,
        system_prompt="sys",
        tools=tools,
        tool_executor=lambda function_name, arguments: "ok",
        max_tool_rounds=3,
    )

    output = agent.run({"problem": "demo"})
    assert output == "tool-result"
    assert llm.chat_with_tools_calls == 1
    assert llm.last_max_tool_rounds == 3



def test_base_agent_returns_final_text_without_tools() -> None:
    llm = _FakeLLMClient()
    agent = BaseAgent(llm_client=llm, system_prompt="sys")

    output = agent.run("only-text")
    assert output == "plain-result"
    assert llm.chat_calls == 1
    assert llm.chat_with_tools_calls == 0
