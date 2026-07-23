import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from omniai.gateway import GatewayRouter
from omniai.gateway.security import WS_POLICY_VIOLATION, TokenBucketRateLimiter
from omniai.protocol import OmniMessage
from omniai.settings import OmniSettings


def make_router(**settings_kwargs) -> GatewayRouter:
    settings = OmniSettings(_env_file=None, api_keys=["good-key"], **settings_kwargs)
    return GatewayRouter(handler=lambda m: m.reply("ok"), settings=settings)


def test_missing_key_rejected():
    client = TestClient(make_router().app)
    resp = client.post("/v1/messages", json={"content": "hi"})
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "unauthorized"
    assert resp.headers["WWW-Authenticate"] == "ApiKey"


def test_wrong_key_rejected_valid_key_accepted():
    client = TestClient(make_router().app)
    assert (
        client.post("/v1/messages", json={"content": "x"}, headers={"X-API-Key": "bad"})
        .status_code
        == 401
    )
    resp = client.post("/v1/messages", json={"content": "x"}, headers={"X-API-Key": "good-key"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "ok"


def test_health_and_metrics_exempt():
    client = TestClient(make_router().app)
    assert client.get("/health").status_code == 200


def test_websocket_requires_key():
    client = TestClient(make_router().app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws"):
            pass
    assert excinfo.value.code == WS_POLICY_VIOLATION

    with client.websocket_connect("/ws", headers={"X-API-Key": "good-key"}) as ws:
        ws.send_json({"content": "ping"})
        assert ws.receive_json()["content"] == "ok"


def test_websocket_query_param_fallback():
    client = TestClient(make_router().app)
    with client.websocket_connect("/ws?api_key=good-key") as ws:
        ws.send_json({"content": "ping"})
        assert ws.receive_json()["content"] == "ok"


def test_fail_closed_without_keys():
    with pytest.raises(RuntimeError, match="OMNIAI_API_KEYS"):
        GatewayRouter(
            handler=lambda m: m.reply("ok"), settings=OmniSettings(_env_file=None)
        )


def test_explicit_opt_out_runs_open():
    settings = OmniSettings(_env_file=None, auth_disabled=True)
    client = TestClient(GatewayRouter(handler=lambda m: m.reply("ok"), settings=settings).app)
    assert client.post("/v1/messages", json={"content": "hi"}).status_code == 200


def test_rate_limit_returns_429_with_retry_after():
    router = make_router(rate_limit_rps=1.0, rate_limit_burst=2)
    client = TestClient(router.app)
    headers = {"X-API-Key": "good-key"}
    codes = [
        client.post("/v1/messages", json={"content": "x"}, headers=headers).status_code
        for _ in range(4)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]
    resp = client.post("/v1/messages", json={"content": "x"}, headers=headers)
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_body_size_cap():
    router = make_router(max_body_bytes=100)
    client = TestClient(router.app)
    resp = client.post(
        "/v1/messages",
        json={"content": "y" * 500},
        headers={"X-API-Key": "good-key"},
    )
    assert resp.status_code == 413


def test_token_bucket_refills():
    limiter = TokenBucketRateLimiter(rate=1000.0, burst=1)
    assert limiter.allow("k") is None
    assert limiter.allow("k") is not None  # bucket drained
    import time

    time.sleep(0.01)  # 1000 rps refills within 10ms
    assert limiter.allow("k") is None


def test_no_settings_stays_open_for_embedding():
    router = GatewayRouter(handler=lambda m: m.reply("ok"))
    client = TestClient(router.app)
    assert client.post("/v1/messages", json={"content": "hi"}).status_code == 200
