# omniai.gateway

## `GatewayRouter`

```python
GatewayRouter(
    handler,                       # (OmniMessage) -> OmniMessage, sync or async
    interceptors=None,             # inbound transforms; raise GuardrailViolation to reject (400)
    observers=None,                # fire-and-forget taps on inbound + outbound
    app=None,                      # bring your own FastAPI
    settings=None,                 # OmniSettings => production mode (see below)
    shutdown_hooks=None,           # callables run on app shutdown
    engine=None, buffer=None,      # wired into readiness + metrics
)
```

Attributes: `app` (FastAPI), `metrics` (in production mode). Methods: `add_interceptor`, `add_observer`, `await dispatch(message)`.

**Routes:** `POST /v1/messages`, `WS /ws`, `POST /discord/webhook`, `GET /health`; production mode adds `GET /metrics`, `GET /health/live`, `GET /health/ready`.

**Production mode** (`settings` provided): validates security fail-closed, then installs — innermost to outermost — body-limit middleware, metrics + request-ID middleware, CORS, and `SecurityMiddleware` (auth + rate limit). Error handlers map `EngineUnavailable` → 503 (+`Retry-After`), `BodyLimitExceeded` → 413, unhandled → sanitized 500.

## Channel adapters (`omniai.gateway.adapters`)

`ChannelAdapter` ABC: `to_omni(payload: dict) -> OmniMessage`, `from_omni(message) -> dict`. Implementations: `RestAdapter`, `WebSocketAdapter`, `DiscordAdapter` (2000-char reply truncation).

## Security (`omniai.gateway.security`)

- `SecurityMiddleware(app, settings, rate_limiter=None)` — API-key auth (`X-API-Key`; WS `api_key` query fallback; close code `WS_POLICY_VIOLATION = 4401`), rate limiting (429 + `Retry-After`), Content-Length cap. Exempt paths: `/health*`, `/metrics`.
- `TokenBucketRateLimiter(rate, burst)` — `allow(key) -> None | retry_after_seconds`; swap in a shared-store implementation for multi-replica.
- `BodyLimitMiddleware(app, settings)` — streams-aware body cap; raises `BodyLimitExceeded` (an `HTTPException` subclass, status 413).

## Observability (`omniai.gateway.observability`)

`Metrics` (per-router Prometheus registry: `omniai_requests_total`, `omniai_request_latency_seconds`, `omniai_engine_tokens_total`, `omniai_breaker_open`, `omniai_learning_cycles_total`), `request_id_var` (contextvar), `configure_logging(settings)`, `setup_tracing(settings)`, `JsonFormatter`.
