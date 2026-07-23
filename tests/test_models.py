import json

import httpx
import pytest
from pydantic import BaseModel

from omniai.engine import ModelEngine
from omniai.graph import tool
from omniai.models import (
    AnthropicChatModel,
    EngineChatModel,
    OpenAIChatModel,
    StructuredOutputError,
    omni_to_openai,
)
from omniai.protocol import OmniMessage, Role, ToolCall


@tool
def get_weather(city: str) -> str:
    """Get current weather."""
    return f"sunny in {city}"


# -- canonical message conversion ------------------------------------------


def test_omni_to_openai_handles_tool_flow():
    call = ToolCall(name="get_weather", arguments={"city": "Paris"})
    messages = omni_to_openai(
        [
            OmniMessage(role=Role.USER, content="weather?"),
            OmniMessage(role=Role.ASSISTANT, content="", tool_calls=[call]),
            OmniMessage(role=Role.TOOL, content="sunny", metadata={"tool_call_id": call.id}),
        ]
    )
    assert messages[0] == {"role": "user", "content": "weather?"}
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert messages[2] == {"role": "tool", "tool_call_id": call.id, "content": "sunny"}


# -- OpenAI adapter --------------------------------------------------------


def _openai_client(handler):
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler)
    )


async def test_openai_generate_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Oslo"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    model = OpenAIChatModel("gpt-test", api_key="k", client=_openai_client(handler))
    result = await model.generate(
        [{"role": "user", "content": "weather in Oslo?"}], tools=[get_weather]
    )
    assert seen["path"].endswith("/chat/completions")
    assert seen["body"]["tools"][0]["function"]["name"] == "get_weather"
    assert result.content == ""
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Oslo"}
    assert result.stop_reason == "tool_calls"
    assert result.usage["prompt_tokens"] == 10


async def test_openai_retries_5xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )

    model = OpenAIChatModel("gpt-test", client=_openai_client(handler), retries=2)
    result = await model.generate([{"role": "user", "content": "x"}])
    assert result.content == "hi"
    assert calls["n"] == 2


# -- Anthropic adapter -----------------------------------------------------


async def test_anthropic_translates_and_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["version"] = request.headers.get("anthropic-version")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Rome"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 12, "output_tokens": 7},
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler)
    )
    model = AnthropicChatModel("claude-sonnet-5", api_key="k", client=client)
    result = await model.generate(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "weather in Rome?"},
        ],
        tools=[get_weather],
    )
    assert seen["path"] == "/v1/messages"
    assert seen["version"]
    assert seen["body"]["system"] == "Be terse."
    assert seen["body"]["messages"] == [{"role": "user", "content": "weather in Rome?"}]
    assert seen["body"]["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assert result.content == "Checking."
    assert result.tool_calls[0].arguments == {"city": "Rome"}
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 7}


async def test_anthropic_tool_results_become_tool_result_blocks():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "It is sunny."}], "usage": {}},
        )

    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler)
    )
    model = AnthropicChatModel("claude-sonnet-5", client=client)
    await model.generate(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "X"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
        ]
    )
    sent = seen["body"]["messages"]
    assert sent[1]["content"][0]["type"] == "tool_use"
    assert sent[2]["role"] == "user"
    assert sent[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "sunny",
    }


# -- Engine adapter --------------------------------------------------------


async def test_engine_chat_model_delegates_to_engine():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "get_weather"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = ModelEngine.create({"model": "m"})
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    model = EngineChatModel(engine)
    result = await model.generate([{"role": "user", "content": "x"}], tools=[get_weather])
    assert result.content == "ok"


# -- structured output -----------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


async def test_structured_output_parses_valid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "JSON Schema" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"name": "Ada", "age": 36}'}}
                ]
            },
        )

    model = OpenAIChatModel("gpt-test", client=_openai_client(handler))
    person = await model.with_structured_output(Person).invoke("Who was Ada Lovelace?")
    assert person == Person(name="Ada", age=36)


async def test_structured_output_retries_with_error_feedback():
    responses = iter(
        [
            "Sure! Here you go: not json at all",
            '```json\n{"name": "Ada", "age": 36}\n```',
        ]
    )
    seen_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": next(responses)}}]},
        )

    model = OpenAIChatModel("gpt-test", client=_openai_client(handler))
    person = await model.with_structured_output(Person, max_retries=1).invoke("go")
    assert person.age == 36
    # Second call must include the error feedback turn.
    assert any("invalid" in m["content"].lower() for m in seen_bodies[1]["messages"])


async def test_structured_output_gives_up_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "nope"}}]}
        )

    model = OpenAIChatModel("gpt-test", client=_openai_client(handler))
    with pytest.raises(StructuredOutputError):
        await model.with_structured_output(Person, max_retries=1).invoke("go")
