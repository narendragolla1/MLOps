# How to monitor with metrics, probes, and logs

All of this activates when the router is built with `settings` (see [auth guide](auth_rate_limiting.md)); none of it requires an API key to scrape.

## Prometheus metrics — `GET /metrics`

| Metric | Labels | Meaning |
| --- | --- | --- |
| `omniai_requests_total` | `method`, `path`, `status` | Request counts. `path` is the **route template**, never the raw URL, so scanner 404s can't explode cardinality (they share `path="unmatched"`). |
| `omniai_request_latency_seconds` | `method`, `path` | Latency histogram. |
| `omniai_engine_tokens_total` | `kind` (`prompt`/`completion`) | Token throughput, fed by engine usage data. |
| `omniai_breaker_open` | — | 1 while the engine circuit breaker is open. |
| `omniai_learning_cycles_total` | `status` | Continuous-learning outcomes (`deployed`/`rejected`/`skipped`). |

## Probes

- `GET /health/live` — process is up (container healthchecks point here).
- `GET /health/ready` — returns 503 with a `problems` list when the database is unreachable, the engine breaker is open, or the engine supervisor has terminally failed (`engine: supervisor failed (...)`). Wire this to your load balancer / orchestrator readiness.

## Request IDs and JSON logs

Every response carries `X-Request-ID` (inbound values are honored, otherwise generated). Logs are single-line JSON with the current request ID injected:

```json
{"ts": "...", "level": "INFO", "logger": "omniai", "message": "...", "request_id": "req_ab12..."}
```

Config: `OMNIAI_LOG_LEVEL` (default INFO), `OMNIAI_LOG_JSON=false` for plain text in dev.

## Traces (OTLP)

Spans already instrument gateway dispatch, every graph node, engine calls (with token counts), LoRA loads, and learning cycles. Without the OTel SDK they land in an in-process recorder (`omniai.telemetry.recorder` — handy in tests). To export for real:

```bash
pip install -e ".[telemetry]"
export OMNIAI_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
export OMNIAI_SERVICE_NAME=omniai-gateway
```

## Learning-cycle metrics

The production app factory wires `learner.on_report` to the `omniai_learning_cycles_total` counter automatically; do the same in custom wiring:

```python
learner.on_report = lambda r: router.metrics.learning_cycles.labels(r["status"]).inc()
```
