import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from omniai.engine import ModelEngine
from omniai.engine.resilience import (
    BreakerState,
    CircuitBreaker,
    EngineSupervisor,
    EngineUnavailable,
    with_retries,
)
from omniai.gateway import GatewayRouter
from omniai.protocol import OmniMessage


# -- retries ---------------------------------------------------------------

async def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    assert await with_retries(flaky, attempts=3, base_delay=0.001) == "ok"
    assert calls["n"] == 3


async def test_no_retry_on_client_error():
    calls = {"n": 0}
    request = httpx.Request("POST", "http://x/")

    async def bad_request():
        calls["n"] += 1
        raise httpx.HTTPStatusError(
            "422", request=request, response=httpx.Response(422, request=request)
        )

    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(bad_request, attempts=3, base_delay=0.001)
    assert calls["n"] == 1  # 4xx is not transient


async def test_exhausted_retries_raise_last_error():
    async def always_down():
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        await with_retries(always_down, attempts=2, base_delay=0.001)


# -- circuit breaker -------------------------------------------------------

async def _failing():
    raise httpx.ConnectError("down")


async def test_breaker_opens_after_threshold_and_recovers():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await breaker.call(_failing)
    assert breaker.state is BreakerState.OPEN
    with pytest.raises(EngineUnavailable):  # fail fast, no call attempted
        await breaker.call(_failing)

    await asyncio.sleep(0.06)  # reset window elapses -> half-open probe

    async def healthy():
        return "up"

    assert await breaker.call(healthy) == "up"
    assert breaker.state is BreakerState.CLOSED


async def test_half_open_failure_reopens():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.02)
    with pytest.raises(httpx.ConnectError):
        await breaker.call(_failing)
    await asyncio.sleep(0.03)
    with pytest.raises(httpx.ConnectError):  # probe fails
        await breaker.call(_failing)
    assert breaker.state is BreakerState.OPEN


# -- engine integration ----------------------------------------------------

def _engine_with_transport(handler, **config):
    engine = ModelEngine.create({"model": "m", "backend": "vllm", **config})
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    return engine


async def test_chat_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    engine = _engine_with_transport(handler, retries=3)
    assert await engine.chat_text([{"role": "user", "content": "x"}]) == "ok"
    assert calls["n"] == 3


async def test_chat_raises_engine_unavailable_when_down():
    engine = _engine_with_transport(lambda request: httpx.Response(500), retries=2)
    with pytest.raises(EngineUnavailable):
        await engine.chat([{"role": "user", "content": "x"}])


async def test_breaker_failfast_after_repeated_failures():
    engine = _engine_with_transport(
        lambda request: httpx.Response(500),
        retries=1,
        breaker_failure_threshold=2,
        breaker_reset_s=60.0,
    )
    for _ in range(2):
        with pytest.raises(EngineUnavailable):
            await engine.chat([{"role": "user", "content": "x"}])
    assert engine.breaker.state is BreakerState.OPEN


def test_unmanaged_mode_uses_external_url_and_spawns_nothing():
    engine = ModelEngine.create(
        {"model": "m", "managed": False, "external_base_url": "http://vllm:8000/"}
    )
    assert engine.config.base_url == "http://vllm:8000"
    assert engine.adapter.process is None


async def test_gateway_maps_engine_unavailable_to_503():
    async def handler(message: OmniMessage) -> OmniMessage:
        raise EngineUnavailable("backend down")

    router = GatewayRouter(handler=handler)
    client = TestClient(router.app, raise_server_exceptions=False)
    resp = client.post("/v1/messages", json={"content": "hi"})
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "engine_unavailable"
    assert "Retry-After" in resp.headers


async def test_unhandled_errors_return_problem_json_without_trace():
    def handler(message: OmniMessage) -> OmniMessage:
        raise RuntimeError("secret internal detail")

    router = GatewayRouter(handler=handler)
    client = TestClient(router.app, raise_server_exceptions=False)
    resp = client.post("/v1/messages", json={"content": "hi"})
    assert resp.status_code == 500
    assert "secret" not in resp.text


# -- supervision -----------------------------------------------------------

class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 1


async def test_supervisor_restarts_and_reapplies_lora():
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)  # crashed
            self.starts = 0

        def start(self):
            self.starts += 1
            self.process = FakeProcess(alive=True)

        async def wait_ready(self, timeout):
            return True

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.active_lora = "skills-v3"
            self.active_lora_path = "/adapters/skills-v3"
            self.lora_loads = []

        async def load_lora_adapter(self, name, path):
            self.lora_loads.append((name, path))

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=3, restart_backoff_base=0.01
    )
    supervisor.start()
    await asyncio.sleep(0.3)
    await supervisor.stop()
    assert engine.adapter.starts >= 1
    assert ("skills-v3", "/adapters/skills-v3") in engine.lora_loads


async def test_supervisor_ignores_healthy_process():
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=True)
            self.starts = 0

        def start(self):
            self.starts += 1

    class FakeEngine:
        adapter = None
        active_lora = None
        active_lora_path = None

    engine = FakeEngine()
    engine.adapter = FakeAdapter()
    supervisor = EngineSupervisor(engine, check_interval=0.01)
    supervisor.start()
    await asyncio.sleep(0.05)
    await supervisor.stop()
    assert engine.adapter.starts == 0
