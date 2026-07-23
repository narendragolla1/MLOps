# omniai.settings

## `OmniSettings`

Pydantic-settings model; every field reads from the environment with the `OMNIAI_` prefix (and a local `.env` file). `get_settings()` returns the cached process-wide instance; `reset_settings_cache()` forces a re-read (tests).

| Env variable | Default | Meaning |
| --- | --- | --- |
| `OMNIAI_DATABASE_URL` | `sqlite+aiosqlite:///interactions.db` | Any SQLAlchemy **async** URL; Postgres (`postgresql+asyncpg://...`) in production. |
| `OMNIAI_API_KEYS` | *(empty)* | Comma-separated API keys. Empty + auth enabled ⇒ startup refusal (fail-closed). |
| `OMNIAI_AUTH_DISABLED` | `false` | Explicit opt-out of auth. Never in production. |
| `OMNIAI_RATE_LIMIT_RPS` | `10.0` | Sustained requests/second per key. |
| `OMNIAI_RATE_LIMIT_BURST` | `20` | Token-bucket capacity. |
| `OMNIAI_CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins; empty = no CORS. |
| `OMNIAI_MAX_BODY_BYTES` | `1000000` | Request body cap (Content-Length + streaming enforcement). |
| `OMNIAI_ENGINE_BASE_URL` | *(unset)* | External serving server URL (e.g. `http://vllm:8000`). |
| `OMNIAI_ENGINE_MANAGED` | `true` | `false` = attach to the external server instead of spawning. |
| `OMNIAI_REQUEST_TIMEOUT_S` | `120.0` | Engine HTTP timeout. |
| `OMNIAI_ENGINE_RETRIES` | `3` | Retry attempts for transient engine failures. |
| `OMNIAI_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures before the breaker opens. |
| `OMNIAI_BREAKER_RESET_S` | `30.0` | Open→half-open window. |
| `OMNIAI_SUPERVISOR_MAX_RESTARTS` | `5` | Subprocess restarts before terminal failure. |
| `OMNIAI_ENGINE_MAX_CONCURRENCY` | `32` | Backpressure semaphore size. |
| `OMNIAI_LOG_LEVEL` | `INFO` | Root log level. |
| `OMNIAI_LOG_JSON` | `true` | Single-line JSON logs (`false` = plain text). |
| `OMNIAI_OTLP_ENDPOINT` | *(unset)* | OTLP traces endpoint; requires the `telemetry` extra. |
| `OMNIAI_SERVICE_NAME` | `omniai-gateway` | OTel service name. |

Methods: `auth_enabled` property, `validate_security()` (raises `RuntimeError` when fail-closed conditions aren't met).

## `omniai.app.create_app`

```python
create_app(settings: OmniSettings | None = None) -> FastAPI
```

The production factory: validates security, builds the engine (external mode from settings), Postgres/SQLite buffer, continuous learner (threshold 1000), guardrails, and a default chat graph, wired into a fully hardened `GatewayRouter`. Container entrypoint: `uvicorn omniai.app:create_app --factory`.
