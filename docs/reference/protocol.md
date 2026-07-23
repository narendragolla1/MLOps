# omniai.protocol

## `OmniMessage`

Pydantic model; the canonical message ([concept](../concepts/messages.md)).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | `str` | `"msg_" + hex` | Unique; used for idempotent persistence. |
| `session_id` | `str` | `"default"` | Conversation grouping key. |
| `channel` | `Channel` | `INTERNAL` | Origin channel. |
| `role` | `Role` | `USER` | |
| `content` | `str` | `""` | |
| `tool_calls` | `list[ToolCall]` | `[]` | Set on assistant messages requesting tools. |
| `metadata` | `dict[str, Any]` | `{}` | Tool results carry `tool_call_id` / `tool_name` here. |
| `created_at` | `datetime` | now (UTC) | |

Methods:

- `reply(content, role=Role.ASSISTANT) -> OmniMessage` — response bound to the same session/channel; sets `metadata["in_reply_to"]`.
- `to_openai() -> dict` — `{"role", "content"}` rendering.

## `Role`

`StrEnum`: `SYSTEM`, `USER`, `ASSISTANT`, `TOOL`.

## `Channel`

`StrEnum`: `REST`, `WEBSOCKET`, `DISCORD`, `INTERNAL`.

## `ToolCall`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | `"call_" + hex` |
| `name` | `str` | required |
| `arguments` | `dict[str, Any]` | `{}` |
