# Chat models

`ChatModel` is the provider-neutral interface every LLM sits behind. Agents, graphs, and structured output are written against it once; swapping GPT for Claude for your own vLLM deployment is a one-line change.

## The interface

```python
class ChatModel(abc.ABC):
    async def generate(self, messages, tools=None, **kwargs) -> ChatResult: ...
    def with_structured_output(self, schema, max_retries=2) -> StructuredOutput: ...
```

- `messages` — the canonical chat wire format (see [messages](messages.md)).
- `tools` — `Tool` objects; adapters translate their schemas natively.
- `ChatResult` — normalized output: `content`, `tool_calls` (as protocol `ToolCall`s with dict arguments), `usage` (`prompt_tokens`/`completion_tokens`), `stop_reason`, and `raw` (the untouched provider response, for anything provider-specific).

## Why an abstraction (and what it must hide)

Providers differ in exactly the places application code shouldn't care about:

| Concern | OpenAI | Anthropic | Self-hosted engine |
| --- | --- | --- | --- |
| System prompt | a `system` message | top-level `system` param | system message |
| Tool schema | `tools[].function.parameters` | `tools[].input_schema` | OpenAI format |
| Tool calls in reply | `message.tool_calls[]` (JSON-string args) | `tool_use` content blocks (dict input) | OpenAI format |
| Tool results | `role: "tool"` message | `tool_result` block in a user turn | OpenAI format |
| Turn structure | free | strict user/assistant alternation | free |

Each adapter owns its translation completely (`omniai/models/anthropic.py` is the instructive one — block conversion plus alternation merging), so none of these differences leak upward.

## Reliability

Cloud adapters wrap calls in `with_retries` (backoff + jitter on connect errors and 5xx). `EngineChatModel` goes further: it delegates to `ModelEngine.chat`, inheriting the circuit breaker, backpressure semaphore, token metrics, and active-LoRA routing described in [serving engines](serving_engines.md).

## Extending

Implementing a new provider is one class with one method — see [How to write a custom chat model](../how_to/custom_chat_model.md).
