# Anthropic

```python
from omniai.models import AnthropicChatModel

model = AnthropicChatModel(
    "claude-sonnet-5",
    api_key="sk-ant-...",
    max_tokens=1024,        # Anthropic requires max_tokens; this is the default
)
result = await model.generate(messages, tools=[...])
```

## Translation details

The adapter converts the canonical chat format to the Messages API transparently:

- **System prompts**: `system`-role messages are extracted and joined into the top-level `system` parameter.
- **Tools**: each `Tool` becomes `{"name", "description", "input_schema": tool.json_schema}`.
- **Assistant tool calls**: `tool_calls` become `tool_use` content blocks (text content preserved as a `text` block alongside).
- **Tool results**: `{"role": "tool", "tool_call_id": ...}` messages become `tool_result` blocks inside a **user** turn.
- **Strict alternation**: consecutive same-role turns are merged into multi-block messages, since the API requires user/assistant alternation — multiple tool results collapse into one user turn automatically.

On the way back, `text` blocks concatenate into `ChatResult.content`, `tool_use` blocks become `ToolCall`s (input is already a dict), and `usage.input_tokens`/`output_tokens` are normalized to `prompt_tokens`/`completion_tokens`.

## Headers

`anthropic-version` and `x-api-key` are set for you — including on injected test clients, so a custom `client=` never silently drops required headers.
