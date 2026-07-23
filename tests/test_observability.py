import json
import logging

import httpx
from fastapi.testclient import TestClient

from omniai.engine import ModelEngine
from omniai.engine.resilience import BreakerState
from omniai.gateway import GatewayRouter
from omniai.gateway.observability import JsonFormatter, request_id_var
from omniai.settings import OmniSettings


def make_router(**kwargs) -> GatewayRouter:
    settings = OmniSettings(_env_file=None, api_keys=["k"])
    return GatewayRouter(handler=lambda m: m.reply("ok"), settings=settings, **kwargs)


AUTH = {"X-API-Key": "k"}


def test_probes_no_auth_required():
    client = TestClient(make_router().app)
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_request_id_echoed_and_generated():
    client = TestClient(make_router().app)
    resp = client.post("/v1/messages", json={"content": "x"}, headers=AUTH)
    assert resp.headers["X-Request-ID"].startswith("req_")
    resp = client.post(
        "/v1/messages", json={"content": "x"}, headers={**AUTH, "X-Request-ID": "trace-42"}
    )
    assert resp.headers["X-Request-ID"] == "trace-42"


def test_metrics_expose_request_counts_and_latency():
    router = make_router()
    client = TestClient(router.app)
    client.post("/v1/messages", json={"content": "x"}, headers=AUTH)
    body = client.get("/metrics").text
    assert 'omniai_requests_total{method="POST",path="/v1/messages",status="200"}' in body
    assert "omniai_request_latency_seconds_bucket" in body


def test_metrics_labels_bounded_for_unmatched_paths():
    """Scanner 404s must not create one Prometheus series per random URL."""
    router = make_router()
    client = TestClient(router.app)
    for i in range(3):
        client.get(f"/wp-admin/exploit-{i}", headers=AUTH)
    body = client.get("/metrics").text
    assert "exploit-0" not in body
    assert 'path="unmatched"' in body


def test_engine_token_usage_feeds_metrics():
    engine = ModelEngine.create({"model": "m"})

    def backend(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(backend)
    )

    async def handler(message):
        await engine.chat([{"role": "user", "content": message.content}])
        return message.reply("done")

    settings = OmniSettings(_env_file=None, api_keys=["k"])
    router = GatewayRouter(handler=handler, settings=settings, engine=engine)
    client = TestClient(router.app)
    client.post("/v1/messages", json={"content": "x"}, headers=AUTH)
    body = client.get("/metrics").text
    assert 'omniai_engine_tokens_total{kind="prompt"} 7.0' in body
    assert 'omniai_engine_tokens_total{kind="completion"} 3.0' in body


def test_breaker_gauge_and_readiness_reflect_open_breaker():
    engine = ModelEngine.create({"model": "m"})
    router = make_router(engine=engine)
    client = TestClient(router.app)

    engine.breaker.state = BreakerState.OPEN
    assert "omniai_breaker_open 1.0" in client.get("/metrics").text
    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert "engine: circuit breaker open" in ready.json()["problems"]

    engine.breaker.state = BreakerState.CLOSED
    assert "omniai_breaker_open 0.0" in client.get("/metrics").text
    assert client.get("/health/ready").status_code == 200


def test_readiness_reports_database_failure():
    class BrokenBuffer:
        async def count(self):
            raise ConnectionError("db down")

    router = make_router(buffer=BrokenBuffer())
    resp = TestClient(router.app).get("/health/ready")
    assert resp.status_code == 503
    assert any(p.startswith("database:") for p in resp.json()["problems"])


def test_json_formatter_includes_request_id():
    token = request_id_var.set("req_test")
    try:
        record = logging.LogRecord("omniai", logging.INFO, __file__, 1, "hello %s", ("x",), None)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello x"
    assert payload["request_id"] == "req_test"
    assert payload["level"] == "INFO"


def test_learning_cycle_metric_hook():
    router = make_router()
    counted = router.metrics.learning_cycles
    counted.labels("deployed").inc()
    body = TestClient(router.app).get("/metrics").text
    assert 'omniai_learning_cycles_total{status="deployed"} 1.0' in body
