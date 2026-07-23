# How to write a custom chat model

Implement `ChatModel` to plug any provider into agents, structured output, and graphs.

## The contract

One async method. Input is the canonical OpenAI-shaped message list; output is a normalized `ChatResult`.

```python
from omniai.models.base import ChatModel, ChatResult
from omniai.protocol import ToolCall

class MyProviderModel(ChatModel):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    async def generate(self, messages, tools=None, **kwargs) -> ChatResult:
        payload = self._translate(messages, tools)     # your provider's wire format
        data = await self._call_api(payload, **kwargs)
        return ChatResult(
            content=data["text"],
            tool_calls=[ToolCall(id=c["id"], name=c["name"], arguments=c["args"])
                        for c in data.get("calls", [])],
            usage={"prompt_tokens": ..., "completion_tokens": ...},
            stop_reason=data.get("finish"),
            raw=data,
        )
```

That's the whole integration — `create_tool_agent(MyProviderModel(...), tools)` and `with_structured_output` now work.

## Rules to follow

- **Honor `tools`.** If your provider has native tool calling, translate `t.to_openai()` / `t.json_schema` per tool; parse its calls back into `ToolCall` objects (arguments as a dict, not a JSON string). If it doesn't, fall back to `render_tool_prompt`.
- **Handle the tool-flow message shapes**: assistant messages with `tool_calls`, and `{"role": "tool", "tool_call_id": ..., "content": ...}` results. See `omniai/models/anthropic.py` for a full translation example (system extraction, block conversion, role-alternation merging).
- **Retry transient failures.** Wrap HTTP calls with `omniai.engine.resilience.with_retries` — the built-in adapters retry connect errors and 5xx with backoff + jitter.
- **Normalize usage** to `prompt_tokens` / `completion_tokens` so metrics stay consistent.

## Testing

Inject an `httpx.AsyncClient` with `httpx.MockTransport` (all built-in adapters accept a `client=` parameter for exactly this) and assert on the request your adapter builds — see `tests/test_models.py` for the pattern.
