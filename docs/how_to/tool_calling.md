# How to do tool calling

## Define a tool

```python
from omniai.graph import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"22 {unit} in {city}"
```

Type hints become the JSON Schema; parameters with defaults become optional; the docstring becomes the description. Override the name/description with `@tool(name=..., description=...)`. Async functions work identically.

## Inspect the schema

```python
get_weather.json_schema          # raw JSON Schema of the parameters
get_weather.to_openai()          # OpenAI function-calling tool definition
```

## Pass tools to a model

```python
result = await model.generate(messages, tools=[get_weather])
for call in result.tool_calls:           # ToolCall(id=..., name=..., arguments={...})
    print(call.name, call.arguments)
```

Every provider adapter translates the schema to its native format (OpenAI `tools`, Anthropic `input_schema`) — see [Integrations](../integrations/index.md).

## Execute a call safely

```python
output = await get_weather.execute(call.arguments)   # validates first
```

`execute` validates (and coerces) arguments against the schema before touching your function; bad arguments raise `ToolValidationError` instead of reaching your code. Accepts a dict or a raw JSON string.

## Let the agent do all of this

For the standard loop — model requests tools, framework executes them, results feed back — use [`create_tool_agent`](../tutorials/build_an_agent.md) instead of hand-rolling.

## Prompt-based fallback

For models without native tool calling, `render_tool_prompt(tools)` (in `omniai.graph.tools`) produces a system-prompt fragment instructing the model to emit `{"tool": ..., "arguments": ...}` JSON.
