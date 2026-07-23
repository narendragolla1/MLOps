# omniai.models

## `ChatModel` (abstract)

```python
async def generate(messages: list[dict], tools: list[Tool] | None = None, **kwargs) -> ChatResult
def with_structured_output(schema: type[BaseModel], max_retries: int = 2) -> StructuredOutput
```

`messages` uses the canonical chat wire format ([concept](../concepts/chat_models.md)). Extra `kwargs` pass through to the provider request (e.g. `temperature`, or `model=` to override per call).

## `ChatResult` (dataclass)

| Field | Type | Notes |
| --- | --- | --- |
| `content` | `str` | Empty string when the reply is tool calls only. |
| `tool_calls` | `list[ToolCall]` | Arguments always dicts. |
| `usage` | `dict[str, int]` | Normalized `prompt_tokens` / `completion_tokens`. |
| `stop_reason` | `str \| None` | Provider finish reason. |
| `raw` | `Any` | Untouched provider response. |

## `OpenAIChatModel`

```python
OpenAIChatModel(model, api_key="", base_url="https://api.openai.com/v1",
                timeout=120.0, retries=3, client=None, **default_params)
```

Works with any OpenAI-compatible endpoint via `base_url`. `default_params` merge into every request. `client=` injects an `httpx.AsyncClient` (auth headers still applied).

## `AnthropicChatModel`

```python
AnthropicChatModel(model, api_key="", base_url="https://api.anthropic.com",
                   max_tokens=1024, timeout=120.0, retries=3, client=None, **default_params)
```

Handles system extraction, tool_use/tool_result translation, and role-alternation merging ([details](../integrations/anthropic.md)).

## `EngineChatModel`

```python
EngineChatModel(engine: ModelEngine)
```

Delegates to `engine.chat`, inheriting breaker/retries/backpressure/metrics/LoRA routing.

## `StructuredOutput`

```python
await StructuredOutput(model, schema, max_retries=2).invoke(messages_or_str, **kwargs) -> schema
```

Raises `StructuredOutputError` after all attempts fail. Usually created via `model.with_structured_output(schema)`.

## Helpers

- `omni_to_openai(messages: list[OmniMessage]) -> list[dict]` — history → wire format (tool flow included).
