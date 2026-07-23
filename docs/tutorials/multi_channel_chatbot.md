# Multi-channel chatbot

Ship one agent to REST, WebSockets, and Discord simultaneously — with guardrails, authentication, and interaction logging. This is the production shape of the [Quickstart](../get_started/quickstart.md).

**Prerequisites:** [Build an Agent](build_an_agent.md) · concepts: [gateway](../concepts/gateway.md), [messages](../concepts/messages.md)

## The pipeline

Every channel follows the same path — adapters only translate formats:

```
native payload → adapter.to_omni → interceptors (guardrails) → handler (your graph)
              → observers (logging) → adapter.from_omni → native response
```

## Wire it up

```python
from omniai.gateway import GatewayRouter
from omniai.graph import create_tool_agent
from omniai.guardrails import PromptGuard
from omniai.memory import InteractionBuffer
from omniai.settings import OmniSettings

agent = create_tool_agent(model, [get_weather])          # from the previous tutorial
buffer = InteractionBuffer("interactions.db")            # logs every message, async

settings = OmniSettings(api_keys=["my-secret-key"])      # normally from OMNIAI_* env vars

router = GatewayRouter(
    handler=agent.as_handler(),
    interceptors=[PromptGuard()],       # blocks prompt injection, strips PII
    observers=[buffer],                 # sees every inbound and outbound message
    settings=settings,                  # enables auth, rate limiting, metrics, probes
    buffer=buffer,                      # lets /health/ready check the database
)
```

Passing `settings` turns on the production stack: fail-closed API-key auth, per-key rate limiting, body-size caps, Prometheus `/metrics`, `/health/live` + `/health/ready`, request-ID correlation, and JSON logs. Omit it only for embedded/test use.

## Talk to it on every channel

```bash
uvicorn: uvicorn my_app:router.app --port 8080

# REST
curl -H "X-API-Key: my-secret-key" localhost:8080/v1/messages -d '{"content": "hi", "session_id": "u1"}'

# WebSocket (same JSON frames; api_key query param works where headers can't)
wscat -c "ws://localhost:8080/ws?api_key=my-secret-key"

# Discord webhook
POST /discord/webhook   # Discord's payload is translated to/from OmniMessage automatically
```

`session_id` groups messages into conversations in the interaction log; the Discord adapter maps the channel ID to it automatically.

## What the guardrails do

`PromptGuard` runs before your graph ever sees a message: known prompt-injection patterns are rejected with a 400, and PII (emails, phone numbers, SSNs, Luhn-valid card numbers) is redacted in place with the hits recorded in `message.metadata["pii_redacted"]`. Tune or extend the patterns — see [How to configure guardrails](../how_to/guardrails.md).

## Next steps

- [Deploy with Docker Compose](../how_to/deploy_docker_compose.md) — the full production stack.
- [Observability](../how_to/observability.md) — metrics, probes, and log correlation you just enabled.
- [Continuous learning](continuous_learning.md) — put that interaction log to work.
