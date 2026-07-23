# Build an Agent

In this tutorial you'll build an agent that can use tools to answer questions, understand what happens when tools fail, and then open up the loop to customize its control flow.

**Prerequisites:** [Installation](../get_started/installation.md) · concepts: [chat models](../concepts/chat_models.md), [tools](../concepts/tools.md)

## Define the tools

```python
from omniai.graph import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22C and sunny in {city}"

@tool
def search_flights(origin: str, destination: str, max_price: int = 500) -> str:
    """Find flights between two cities under a price limit."""
    return f"3 flights {origin}->{destination} under ${max_price}"
```

Each decorated function becomes a `Tool` carrying an OpenAI-compatible JSON Schema derived from its signature (`get_weather.to_openai()` shows it). Defaults become optional parameters; docstrings become tool descriptions the model sees.

## Pick a model

```python
from omniai.models import AnthropicChatModel
model = AnthropicChatModel("claude-sonnet-5", api_key="...")
```

Any [`ChatModel`](../reference/models.md) works — the agent is provider-agnostic.

## Create the agent

```python
from omniai.graph import create_tool_agent
from omniai.protocol import OmniMessage

agent = create_tool_agent(
    model,
    [get_weather, search_flights],
    system_prompt="You are a travel assistant. Use tools for live data.",
    max_steps=8,
)

final = await agent.ainvoke({"messages": [OmniMessage(content="Cheap flight from Oslo to Lisbon, and what's the weather there?")]})
print(final.messages[-1].content)
```

Under the hood the compiled graph loops: **model → (tool calls?) → execute tools → append results → model**, until the model answers without tool calls or `max_steps` model turns elapse. Inspect `final.messages` to see the full trajectory, including `Role.TOOL` messages with each tool's output.

## What happens when tools fail

Three failure modes are handled without crashing the run — the error is fed back to the model as tool output so it can self-correct:

- **Invalid arguments** (fails schema validation): the model sees `error: invalid arguments: ...` and can retry with fixed arguments.
- **Unknown tool**: `error: unknown tool 'x'`.
- **Tool raised**: `error: RuntimeError: ...`.

This "errors as observations" behavior is what makes agent loops robust; see the [agents concept](../concepts/agents.md) for the reasoning.

## Customize the control flow

`create_tool_agent` is ~60 lines over the public graph API — when you outgrow it, build the loop yourself:

```python
from omniai.graph import Graph, State, START, END
from omniai.models.base import omni_to_openai
from omniai.protocol import Role

class TravelState(State):
    budget: int = 500

graph = Graph(TravelState)

async def plan(state: TravelState):
    result = await model.generate(omni_to_openai(state.messages), tools=[search_flights])
    reply = OmniMessage(role=Role.ASSISTANT, content=result.content, tool_calls=result.tool_calls)
    return {"messages": [reply]}

graph.add_node("plan", plan)
graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", lambda s: "act" if s.messages[-1].tool_calls else END)
# ... add your own "act" node, approval steps, budget checks, etc.
```

See [How to build graphs](../how_to/graphs_and_edges.md) for conditional edges, cycles, and custom state.

## Next steps

- Serve this agent over HTTP/WebSockets: [Multi-channel chatbot](multi_channel_chatbot.md).
- Force typed answers: [structured output](../how_to/structured_output.md).
- Reuse prompts across agents: [prompt templates](../how_to/prompt_templates.md).
