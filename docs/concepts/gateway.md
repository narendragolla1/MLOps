# Gateway

`GatewayRouter` is the control plane: one FastAPI app serving the same handler over every channel, with a middleware pipeline that keeps cross-cutting concerns out of your graph code.

## Channels and adapters

A `ChannelAdapter` is a pure codec — `to_omni(payload)` and `from_omni(message)`, no I/O — which keeps routes thin and translation trivially testable:

| Route | Adapter | Native format |
| --- | --- | --- |
| `POST /v1/messages` | `RestAdapter` | `{"content", "session_id", "metadata"}` JSON |
| `WS /ws` | `WebSocketAdapter` | same JSON, frame per message |
| `POST /discord/webhook` | `DiscordAdapter` | Discord payloads (author/channel mapped to session; replies truncated to Discord's 2000-char limit) |

Adding a channel (Slack, SMS, …) is one adapter class plus one route.

## The pipeline

```
to_omni → interceptors → observers(in) → handler → observers(out) → from_omni
```

- **Interceptors** transform or reject inbound messages (guardrails live here; raise `GuardrailViolation` → 400). They compose in order.
- **Observers** are fire-and-forget taps on both directions — interaction logging, analytics. They can't alter the message.
- The **handler** is any `OmniMessage → OmniMessage` callable; `compiled_graph.as_handler()` is the usual one.

## Two construction modes

- `GatewayRouter(handler=...)` — open; for embedding and tests.
- `GatewayRouter(handler=..., settings=OmniSettings(...))` — production: validates security config **fail-closed**, then installs auth, rate limiting, body caps, CORS, metrics, probes, request IDs, and JSON logging. `omniai.app.create_app` builds this shape from the environment.

## Errors are structured

Exception handlers translate failures to problem-details JSON — `EngineUnavailable` → 503 + `Retry-After`, body-limit violations → 413, anything unhandled → a generic 500 that **never leaks stack traces or internal messages**.
