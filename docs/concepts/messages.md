# Messages

Everything in OmniAI exchanges one message type: **`OmniMessage`**. Gateways normalize inbound payloads into it at the edge; graphs accumulate it as history; memory persists it; adapters re-encode it per channel on the way out.

## The model

```python
OmniMessage(
    id="msg_...",                 # unique, auto-generated
    session_id="user-42",         # conversation grouping key
    channel=Channel.REST,         # rest | websocket | discord | internal
    role=Role.USER,               # system | user | assistant | tool
    content="What's the weather?",
    tool_calls=[ToolCall(...)],   # set on assistant messages requesting tools
    metadata={...},               # channel/user extras; tool results carry tool_call_id here
    created_at=datetime(...),
)
```

Useful methods: `message.reply(content)` builds a response bound to the same session and channel (with `in_reply_to` metadata); `message.to_openai()` renders the simple role/content dict.

## Roles in the tool flow

A complete tool interaction is three messages:

1. `USER` — the question.
2. `ASSISTANT` with `tool_calls=[ToolCall(name=..., arguments={...})]` — the model's request.
3. `TOOL` with the output as `content` and `metadata={"tool_call_id": ..., "tool_name": ...}` — the observation fed back.

## The wire format

Providers don't speak `OmniMessage` — they speak chat dicts. The canonical wire shape is OpenAI's (`{"role": ..., "content": ...}`, assistant `tool_calls`, `{"role": "tool", "tool_call_id": ...}`), and `omni_to_openai(messages)` converts a history losslessly. Provider adapters translate *from* that canonical shape to their native format (see [chat models](chat_models.md)) — so the rest of the system never contains provider-specific message code.

## Sessions

`session_id` is the unit of conversation: the interaction buffer indexes on it, training-pair extraction groups by it, and channel adapters map their native notion onto it (e.g. the Discord adapter uses the Discord channel ID).
