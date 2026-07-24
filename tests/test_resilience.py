import asyncio
import time

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


async def test_supervisor_failure_is_recorded_not_silent():
    class DeadAdapter:
        process = FakeProcess(alive=False)

        def start(self):
            raise RuntimeError("cannot spawn")

    class FakeEngine:
        adapter = DeadAdapter()
        active_lora = None
        active_lora_path = None

    supervisor = EngineSupervisor(
        FakeEngine(), check_interval=0.01, max_restarts=0, restart_backoff_base=0.01
    )
    supervisor.start()
    await asyncio.sleep(0.1)
    assert supervisor.failed
    assert "max_restarts" in (supervisor.failure_reason or "")
    await supervisor.stop()


async def test_semaphore_bounds_engine_concurrency():
    active = {"now": 0, "max": 0}

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    engine = _engine_with_transport(slow_handler, max_concurrent_requests=1)
    await asyncio.gather(*(engine.chat([{"role": "user", "content": "x"}]) for _ in range(3)))
    assert active["max"] == 1  # requests serialized by the backpressure semaphore
    assert engine.in_flight == 0


async def test_half_open_allows_single_probe():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
    with pytest.raises(httpx.ConnectError):
        await breaker.call(_failing)
    await asyncio.sleep(0.02)  # breaker moves to half-open

    started = asyncio.Event()

    async def slow_probe():
        started.set()
        await asyncio.sleep(0.05)
        return "up"

    probe = asyncio.create_task(breaker.call(slow_probe))
    await started.wait()
    with pytest.raises(EngineUnavailable, match="probe in flight"):
        await breaker.call(slow_probe)  # concurrent call fails fast
    assert await probe == "up"
    assert breaker.state is BreakerState.CLOSED


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


# -- EngineSupervisor extended tests ----------------------------------------


async def test_supervisor_process_alive_checks():
    """Verify _process_alive correctly detects process state."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=True)

        def start(self):
            self.process = FakeProcess(alive=True)

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()

    engine = FakeEngine()
    supervisor = EngineSupervisor(engine)

    # Process alive
    assert supervisor._process_alive()

    # Process dead
    engine.adapter.process.alive = False
    assert not supervisor._process_alive()

    # No process
    engine.adapter.process = None
    assert not supervisor._process_alive()


async def test_supervisor_tracks_restart_count():
    """Verify supervisor increments restart counter."""
    class CountingAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)
            self.start_count = 0

        def start(self):
            self.start_count += 1
            self.process = FakeProcess(alive=True)

        async def wait_ready(self, timeout):
            return True

    class FakeEngine:
        def __init__(self):
            self.adapter = CountingAdapter()
            self.active_lora = None
            self.active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=3, restart_backoff_base=0.001
    )
    supervisor.start()
    await asyncio.sleep(0.2)
    await supervisor.stop()

    assert supervisor.restarts >= 1
    assert supervisor.restarts <= supervisor.max_restarts
    assert engine.adapter.start_count >= supervisor.restarts


async def test_supervisor_enforces_max_restarts_limit():
    """Verify supervisor fails when max_restarts is exceeded."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)  # always dead

        def start(self):
            self.process = FakeProcess(alive=False)  # stays dead

        async def wait_ready(self, timeout):
            return False  # never ready

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.active_lora = None
            self.active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=2, restart_backoff_base=0.001
    )
    supervisor.start()
    await asyncio.sleep(0.2)

    assert supervisor.failed
    assert "max_restarts" in (supervisor.failure_reason or "")
    await supervisor.stop()


async def test_supervisor_without_active_lora():
    """Verify supervisor handles case where no LoRA is active."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)

        def start(self):
            self.process = FakeProcess(alive=True)

        async def wait_ready(self, timeout):
            return True

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.active_lora = None
            self.active_lora_path = None
            self.lora_loads = []

        async def load_lora_adapter(self, name, path):
            self.lora_loads.append((name, path))

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=3, restart_backoff_base=0.01
    )
    supervisor.start()
    await asyncio.sleep(0.15)
    await supervisor.stop()

    assert engine.adapter.process.alive
    assert len(engine.lora_loads) == 0  # No LoRA to load


async def test_supervisor_wait_ready_timeout():
    """Verify supervisor retries when wait_ready times out."""
    class SlowAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)
            self.wait_ready_calls = 0

        def start(self):
            self.process = FakeProcess(alive=True)
            self.wait_ready_calls = 0

        async def wait_ready(self, timeout):
            self.wait_ready_calls += 1
            return False  # Never ready

    class FakeEngine:
        def __init__(self):
            self.adapter = SlowAdapter()
            self.active_lora = None
            self.active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.02, max_restarts=2, restart_backoff_base=0.001
    )
    supervisor.start()
    await asyncio.sleep(0.15)

    # Process should have been detected as dead and restarted
    assert supervisor.restarts >= 1
    await supervisor.stop()


async def test_supervisor_graceful_stop_cancels_watch():
    """Verify supervisor.stop() cancels the watch task."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=True)

    class FakeEngine:
        adapter = FakeAdapter()
        active_lora = None
        active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(engine, check_interval=0.01)

    assert supervisor._task is None
    supervisor.start()
    assert supervisor._task is not None
    task = supervisor._task

    await supervisor.stop()
    assert supervisor._task is None
    assert task.cancelled() or task.done()


async def test_supervisor_multiple_restarts_with_backoff():
    """Verify supervisor applies backoff between restarts."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)
            self.start_times = []

        def start(self):
            self.start_times.append(time.monotonic())
            self.process = FakeProcess(alive=True)

        async def wait_ready(self, timeout):
            # Die immediately so supervisor tries restart again
            self.process = FakeProcess(alive=False)
            return True

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.active_lora = None
            self.active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=2, restart_backoff_base=0.02
    )
    supervisor.start()
    await asyncio.sleep(0.2)
    await supervisor.stop()

    assert len(engine.adapter.start_times) >= 2
    # Check that there's a gap between restart attempts (backoff)
    if len(engine.adapter.start_times) >= 2:
        gap = engine.adapter.start_times[1] - engine.adapter.start_times[0]
        assert gap > 0.01  # Should have some backoff


async def test_supervisor_start_is_idempotent():
    """Verify calling start() multiple times doesn't create multiple tasks."""
    class FakeAdapter:
        process = FakeProcess(alive=True)

    class FakeEngine:
        adapter = FakeAdapter()
        active_lora = None
        active_lora_path = None

    supervisor = EngineSupervisor(FakeEngine(), check_interval=0.01)

    supervisor.start()
    task1 = supervisor._task
    supervisor.start()
    task2 = supervisor._task

    assert task1 is task2  # Same task
    await supervisor.stop()


async def test_supervisor_stop_without_start():
    """Verify supervisor.stop() is safe to call without start()."""
    class FakeAdapter:
        process = FakeProcess(alive=True)

    class FakeEngine:
        adapter = FakeAdapter()
        active_lora = None
        active_lora_path = None

    supervisor = EngineSupervisor(FakeEngine())
    # Should not raise
    await supervisor.stop()
    assert supervisor._task is None


async def test_supervisor_exception_in_restart_sets_failed():
    """Verify supervisor sets failed flag when restart fails."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)

        def start(self):
            raise RuntimeError("cannot start")

        async def wait_ready(self, timeout):
            return False

    class FakeEngine:
        adapter = FakeAdapter()
        active_lora = None
        active_lora_path = None

    supervisor = EngineSupervisor(
        FakeEngine(), check_interval=0.01, max_restarts=1, restart_backoff_base=0.001
    )
    supervisor.start()
    await asyncio.sleep(0.1)

    assert supervisor.failed
    assert supervisor.failure_reason is not None
    assert "cannot start" in supervisor.failure_reason
    await supervisor.stop()


async def test_supervisor_lora_reapply_called_after_restart():
    """Verify supervisor reapplies LoRA after successful restart."""
    lora_calls = []

    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=False)
            self.restart_count = 0

        def start(self):
            self.restart_count += 1
            self.process = FakeProcess(alive=True)

        async def wait_ready(self, timeout):
            return True

    class FakeEngine:
        def __init__(self):
            self.adapter = FakeAdapter()
            self.active_lora = "adapter-v2"
            self.active_lora_path = "/path/to/adapter-v2"

        async def load_lora_adapter(self, name, path):
            lora_calls.append((name, path))

    engine = FakeEngine()
    supervisor = EngineSupervisor(
        engine, check_interval=0.01, max_restarts=2, restart_backoff_base=0.01
    )
    supervisor.start()
    await asyncio.sleep(0.15)
    await supervisor.stop()

    assert ("adapter-v2", "/path/to/adapter-v2") in lora_calls


async def test_supervisor_stopped_event_signal():
    """Verify supervisor respects _stopped event."""
    class FakeAdapter:
        def __init__(self):
            self.process = FakeProcess(alive=True)

    class FakeEngine:
        adapter = FakeAdapter()
        active_lora = None
        active_lora_path = None

    engine = FakeEngine()
    supervisor = EngineSupervisor(engine, check_interval=0.01)
    supervisor.start()
    assert not supervisor._stopped.is_set()

    await supervisor.stop()
    assert supervisor._stopped.is_set()
