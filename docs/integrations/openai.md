# OpenAI

```python
from omniai.models import OpenAIChatModel

model = OpenAIChatModel(
    "gpt-4o-mini",
    api_key="sk-...",
    # base_url="https://api.openai.com/v1",   # default
    temperature=0.2,        # any extra kwargs become default request params
)
result = await model.generate([{"role": "user", "content": "hi"}], tools=[...])
```

## Capabilities

- **Tool calling** — `Tool.to_openai()` definitions are sent as `tools`; `message.tool_calls` are parsed back into `ToolCall` objects (JSON-string arguments decoded to dicts; undecodable arguments preserved under `__unparsed__` rather than dropped).
- **Usage/stop reason** — `usage.prompt_tokens` / `completion_tokens` and `finish_reason` map directly onto `ChatResult`.
- **Retries** — connect errors and 5xx retry with backoff + jitter (`retries=3` default).

## OpenAI-compatible endpoints

The same class talks to anything speaking the Chat Completions protocol — point `base_url` at it:

```python
OpenAIChatModel("llama-3-8b", base_url="http://localhost:8000/v1")        # local vLLM/SGLang
OpenAIChatModel("qwen2.5", base_url="http://localhost:11434/v1")          # Ollama
```

For a vLLM/SGLang server whose *lifecycle you also want managed* (supervision, LoRA hot-swap, breaker), prefer [`EngineChatModel`](self_hosted.md).

## Testing

Pass `client=httpx.AsyncClient(transport=httpx.MockTransport(handler))` to run against a mock; auth headers are still applied to injected clients.
