# How to configure auth and rate limiting

The gateway's security stack activates when you construct `GatewayRouter(..., settings=OmniSettings(...))` — normally sourced from `OMNIAI_*` environment variables.

## API keys (fail-closed)

```bash
export OMNIAI_API_KEYS=key-one,key-two
```

Clients authenticate with the `X-API-Key` header; WebSocket clients may use an `api_key` query parameter (browsers can't set custom WS headers). Keys are compared in constant time. Failures: HTTP 401 with a problem-details body; WebSocket close code **4401** at handshake.

**Fail-closed:** if auth is enabled and no keys are configured, the app **refuses to start**. Running open requires the explicit `OMNIAI_AUTH_DISABLED=true` — there is no accidental unauthenticated deployment.

`/health`, `/health/live`, `/health/ready`, and `/metrics` are exempt so probes and scrapers don't need keys.

## Rate limiting

A token bucket per API key:

```bash
export OMNIAI_RATE_LIMIT_RPS=10      # sustained refill rate
export OMNIAI_RATE_LIMIT_BURST=20    # bucket capacity
```

Over-limit requests get HTTP 429 with a `Retry-After` header (WebSocket: close 4401). The limiter is in-process; behind multiple replicas, effective limits multiply — the `TokenBucketRateLimiter` interface (`allow(key) -> retry_after | None`) is deliberately minimal so a Redis-backed implementation can drop in via `SecurityMiddleware(rate_limiter=...)`.

## Body-size caps

```bash
export OMNIAI_MAX_BODY_BYTES=1000000
```

Enforced twice: a fast `Content-Length` rejection, plus a streaming guard that counts bytes as the body is read — so **chunked uploads without a Content-Length can't bypass the cap**. Either path yields HTTP 413.

## CORS

```bash
export OMNIAI_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

Unset means no cross-origin access (the safe default).

## Related

- [Security concepts](../concepts/security.md) — the full threat model.
- [Settings reference](../reference/settings.md) — every variable.
