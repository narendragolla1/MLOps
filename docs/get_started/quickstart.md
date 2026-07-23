# Quickstart

This page takes you from an installed package to a working tool-calling agent with structured output. Everything here runs against any provider — swap the model class and nothing else changes.

## 1. Pick a chat model

```python
from omniai.models import OpenAIChatModel, AnthropicChatModel

model = AnthropicChatModel("claude-sonnet-5", api_key="...")
# or: model = OpenAIChatModel("gpt-4o-mini", api_key="...")
# or self-hosted: EngineChatModel(ModelEngine.create({...})) — see the integrations docs
```

## 2. Define a tool

The `@tool` decorator turns a typed Python function into a schema-validated tool — the JSON Schema is generated from the type hints, and LLM-produced arguments are validated before the function runs.

```python
from omniai.graph import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22C and sunny in {city}"
```

## 3. Run a prebuilt agent

`create_tool_agent` gives you the standard agent loop as a compiled graph: the model is called with your tool schemas; tool calls are executed and fed back; the loop repeats until the model produces a final answer.

```python
import asyncio
from omniai.graph import create_tool_agent
from omniai.protocol import OmniMessage

agent = create_tool_agent(model, [get_weather], system_prompt="Use tools for live data.")

async def main():
    final = await agent.ainvoke({"messages": [OmniMessage(content="Weather in Lisbon?")]})
    print(final.messages[-1].content)

asyncio.run(main())
```

## 4. Get structured output

```python
from pydantic import BaseModel

class TripPlan(BaseModel):
    city: str
    activity: str
    packing_list: list[str]

plan = await model.with_structured_output(TripPlan).invoke("Plan a sunny afternoon in Lisbon.")
print(plan.packing_list)
```

The wrapper injects the JSON Schema, validates the reply, and on failure feeds the error back to the model and retries — you always get a `TripPlan` instance or a typed error.

## 5. Serve it over HTTP

```python
from omniai.gateway import GatewayRouter
import uvicorn

router = GatewayRouter(handler=agent.as_handler())
uvicorn.run(router.app, port=8080)
# curl localhost:8080/v1/messages -d '{"content": "Weather in Lisbon?"}'
```

The same handler also serves WebSocket (`/ws`) and Discord webhook (`/discord/webhook`) traffic — see the [gateway concept](../concepts/gateway.md).

A complete runnable version of this flow is in [`examples/tool_agent.py`](../../examples/tool_agent.py).

## Next steps

- [Build an Agent](../tutorials/build_an_agent.md) — the full tutorial with error handling and custom graphs.
- [Auth & rate limiting](../how_to/auth_rate_limiting.md) — before exposing that HTTP endpoint anywhere real.
- [Conceptual guide](../concepts/index.md) — how the pieces fit together.
