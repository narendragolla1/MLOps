import pytest

from omniai.graph import create_tool_agent, tool
from omniai.graph.graph import GraphError
from omniai.models.base import ChatModel, ChatResult
from omniai.protocol import OmniMessage, Role, ToolCall


@tool
def get_weather(city: str) -> str:
    """Get current weather."""
    return f"22C in {city}"


@tool
def failing_tool(x: int) -> str:
    """Always breaks."""
    raise RuntimeError("boom")


class ScriptedModel(ChatModel):
    """Plays back canned ChatResults; records every generate() call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def generate(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        return self.results.pop(0)


def _tool_call(name, args):
    return ChatResult(content="", tool_calls=[ToolCall(name=name, arguments=args)])


async def test_agent_executes_tools_then_answers():
    model = ScriptedModel(
        [
            _tool_call("get_weather", {"city": "Paris"}),
            ChatResult(content="It is 22C in Paris."),
        ]
    )
    agent = create_tool_agent(model, [get_weather], system_prompt="Be helpful.")
    final = await agent.ainvoke({"messages": [OmniMessage(content="weather in Paris?")]})

    assert final.messages[-1].content == "It is 22C in Paris."
    # Tool output was appended and fed back to the model.
    tool_messages = [m for m in final.messages if m.role is Role.TOOL]
    assert tool_messages[0].content == "22C in Paris"
    second_call = model.calls[1]["messages"]
    assert any(m["role"] == "tool" and "22C" in m["content"] for m in second_call)
    # System prompt injected once, tools passed natively.
    assert model.calls[0]["messages"][0] == {"role": "system", "content": "Be helpful."}
    assert model.calls[0]["tools"] == [get_weather]


async def test_agent_feeds_tool_errors_back_to_model():
    model = ScriptedModel(
        [
            _tool_call("failing_tool", {"x": 1}),
            ChatResult(content="The tool failed."),
        ]
    )
    agent = create_tool_agent(model, [failing_tool])
    final = await agent.ainvoke({"messages": [OmniMessage(content="go")]})
    tool_message = next(m for m in final.messages if m.role is Role.TOOL)
    assert "error" in tool_message.content
    assert "boom" in tool_message.content
    assert final.messages[-1].content == "The tool failed."


async def test_agent_validates_arguments_and_reports():
    model = ScriptedModel(
        [
            _tool_call("get_weather", {}),  # missing required city
            ChatResult(content="I need a city."),
        ]
    )
    agent = create_tool_agent(model, [get_weather])
    final = await agent.ainvoke({"messages": [OmniMessage(content="weather?")]})
    tool_message = next(m for m in final.messages if m.role is Role.TOOL)
    assert "invalid arguments" in tool_message.content


async def test_agent_handles_unknown_tool():
    model = ScriptedModel(
        [
            _tool_call("nonexistent", {"a": 1}),
            ChatResult(content="done"),
        ]
    )
    agent = create_tool_agent(model, [get_weather])
    final = await agent.ainvoke({"messages": [OmniMessage(content="go")]})
    tool_message = next(m for m in final.messages if m.role is Role.TOOL)
    assert "unknown tool" in tool_message.content


async def test_agent_stops_at_max_steps():
    endless = [_tool_call("get_weather", {"city": "X"}) for _ in range(10)]
    model = ScriptedModel(endless)
    agent = create_tool_agent(model, [get_weather], max_steps=2)
    final = await agent.ainvoke({"messages": [OmniMessage(content="loop forever")]})
    # Stops after 2 model turns even though the model kept requesting tools.
    assert len(model.calls) == 2
    assert final.steps == 2


async def test_agent_multiple_tool_calls_in_one_turn():
    model = ScriptedModel(
        [
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(name="get_weather", arguments={"city": "Paris"}),
                    ToolCall(name="get_weather", arguments={"city": "Oslo"}),
                ],
            ),
            ChatResult(content="Both sunny."),
        ]
    )
    agent = create_tool_agent(model, [get_weather])
    final = await agent.ainvoke({"messages": [OmniMessage(content="compare")]})
    tool_messages = [m for m in final.messages if m.role is Role.TOOL]
    assert {m.content for m in tool_messages} == {"22C in Paris", "22C in Oslo"}


async def test_agent_session_id_propagates():
    model = ScriptedModel([ChatResult(content="hi")])
    agent = create_tool_agent(model, [])
    final = await agent.ainvoke({"messages": [OmniMessage(content="hello", session_id="sess-9")]})
    assert final.messages[-1].session_id == "sess-9"


def test_graph_error_type_still_exported():
    assert issubclass(GraphError, Exception)


async def test_agent_as_gateway_handler():
    """The compiled agent plugs straight into the GatewayRouter."""
    from fastapi.testclient import TestClient

    from omniai.gateway import GatewayRouter

    model = ScriptedModel(
        [
            _tool_call("get_weather", {"city": "Rome"}),
            ChatResult(content="22C in Rome today."),
        ]
    )
    agent = create_tool_agent(model, [get_weather])
    router = GatewayRouter(handler=agent.as_handler())
    client = TestClient(router.app)
    resp = client.post("/v1/messages", json={"content": "weather in Rome?"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "22C in Rome today."


@pytest.mark.parametrize("bad_steps", [0])
async def test_agent_zero_steps_answers_nothing(bad_steps):
    model = ScriptedModel([_tool_call("get_weather", {"city": "X"})])
    agent = create_tool_agent(model, [get_weather], max_steps=bad_steps)
    final = await agent.ainvoke({"messages": [OmniMessage(content="go")]})
    assert final.steps == 1  # one model turn happens, loop then halts
