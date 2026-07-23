"""Self-hosting features: env plumbing, LoRA lifecycle, retries, supervision."""

import asyncio

import httpx
import pytest

from omniai.engine import (
    ADAPTERS,
    EngineConfig,
    EngineSupervisor,
    LoRARegistry,
    ModelEngine,
    RetryPolicy,
    SGLangAdapter,
    VLLMAdapter,
    register_backend,
)


# -- environment / placement -------------------------------------------------

def test_vllm_env_enables_runtime_lora_and_places_gpus():
    config = EngineConfig(model="m", backend="vllm", devices=[2, 3], tensor_parallel_size=2)
    env = VLLMAdapter(config).build_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_user_env_overrides_backend_env():
    config = EngineConfig(model="m", backend="vllm", env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "0"})
    assert VLLMAdapter(config).build_env()["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "0"


def test_devices_must_cover_tensor_parallelism():
    with pytest.raises(ValueError, match="devices"):
        EngineConfig(model="m", tensor_parallel_size=2, devices=[0])


# -- command mapping ---------------------------------------------------------

def test_vllm_prefix_caching_and_lora_slots():
    cmd = VLLMAdapter(EngineConfig(model="m", backend="vllm", max_loras=8)).build_command()
    assert "--enable-prefix-caching" in cmd
    assert ("--max-loras", "8") == tuple(cmd[cmd.index("--max-loras"):][:2])
    cmd = VLLMAdapter(EngineConfig(model="m", backend="vllm", prefix_caching=False)).build_command()
    assert "--enable-prefix-caching" not in cmd


def test_sglang_prefix_caching_opt_out_and_lora_slots():
    cmd = SGLangAdapter(EngineConfig(model="m", backend="sglang", max_loras=2)).build_command()
    assert "--disable-radix-cache" not in cmd
    assert ("--max-loras-per-batch", "2") == tuple(cmd[cmd.index("--max-loras-per-batch"):][:2])
    cmd = SGLangAdapter(
        EngineConfig(model="m", backend="sglang", prefix_caching=False)
    ).build_command()
    assert "--disable-radix-cache" in cmd


def test_custom_backend_registration():
    class FakeAdapter(VLLMAdapter):
        pass

    register_backend("fake", FakeAdapter)
    try:
        engine = ModelEngine.create({"model": "m", "backend": "fake"})
        assert isinstance(engine.adapter, FakeAdapter)
        assert engine.config.backend_name == "fake"
    finally:
        ADAPTERS.pop("fake")


# -- client-side resilience --------------------------------------------------

def _engine_with_handler(handler, **config) -> ModelEngine:
    engine = ModelEngine.create(
        {"model": "m", "backend": "vllm", **config},
        retry=RetryPolicy(attempts=3, backoff_base=0.0, backoff_max=0.0, jitter=0.0),
    )
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    return engine


async def test_chat_retries_transient_errors_until_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    engine = _engine_with_handler(handler)
    assert await engine.chat_text([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["n"] == 3


async def test_chat_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400)

    engine = _engine_with_handler(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


async def test_retries_exhausted_raises_last_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    engine = _engine_with_handler(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.chat([{"role": "user", "content": "hi"}])


# -- LoRA lifecycle ----------------------------------------------------------

async def test_lora_eviction_when_slots_full():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    engine = _engine_with_handler(handler, max_loras=2)
    await engine.load_lora_adapter("v1", "/a/v1")
    await engine.load_lora_adapter("v2", "/a/v2")
    # Slots full; v1 is the rollback target and v2 is active -> nothing
    # evictable, load proceeds and lets the server enforce its own cap.
    await engine.load_lora_adapter("v3", "/a/v3")
    # Now v1 is neither active (v3) nor previous (v2) -> evicted for v4.
    await engine.load_lora_adapter("v4", "/a/v4")
    unloads = [body for path, body in requests if "unload" in path]
    assert unloads == [{"lora_name": "v1"}]
    assert engine.active_lora == "v4"
    assert engine.lora.previous == "v3"


async def test_rollback_reactivates_previous_adapter():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    engine = _engine_with_handler(handler)
    await engine.load_lora_adapter("v1", "/a/v1")
    await engine.load_lora_adapter("v2", "/a/v2")
    assert engine.active_lora == "v2"
    assert await engine.rollback_lora() == "v1"
    assert engine.active_lora == "v1"
    # Roll forward again — v2 stayed loaded.
    assert await engine.rollback_lora() == "v2"


async def test_reapply_active_lora_after_restart():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    engine = _engine_with_handler(handler)
    await engine.load_lora_adapter("v1", "/a/v1")
    assert await engine.reapply_active_lora()
    assert requests.count("/v1/load_lora_adapter") == 2


def test_registry_persists_and_restores(tmp_path):
    path = tmp_path / "registry.json"
    registry = LoRARegistry(persist_path=path)
    registry.register("v1", "/a/v1")
    registry.register("v2", "/a/v2")
    registry.activate("v1")
    registry.activate("v2")

    restored = LoRARegistry(persist_path=path)
    assert set(restored.loaded) == {"v1", "v2"}
    assert restored.active == "v2"
    assert restored.previous == "v1"


# -- supervision -------------------------------------------------------------

class FakeAdapterProc:
    def __init__(self, ready: bool = True):
        self.ready = ready
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def is_alive(self):
        return True

    async def wait_ready(self, timeout: float = 300.0):
        return self.ready


class FakeEngine:
    def __init__(self, ready: bool = True):
        self.adapter = FakeAdapterProc(ready=ready)
        self.reapplied = 0
        self.warmed = 0

    async def health(self):
        return {"process": True, "server": True, "active_lora": None}

    async def reapply_active_lora(self):
        self.reapplied += 1
        return True

    async def warmup(self):
        self.warmed += 1
        return True


async def test_supervisor_restarts_unhealthy_engine_and_reapplies_lora():
    engine = FakeEngine()
    healthy = {"value": False}
    events = []

    async def probe():
        return healthy["value"]

    supervisor = EngineSupervisor(
        engine,
        check_interval=0.01,
        probe=probe,
        on_event=lambda name, data: events.append(name),
    )
    await supervisor.start()
    await asyncio.sleep(0.05)
    healthy["value"] = True  # recovered server now reports healthy
    await asyncio.sleep(0.05)
    await supervisor.stop()

    assert supervisor.restarts >= 1
    assert engine.adapter.starts >= 1
    assert engine.reapplied >= 1
    assert engine.warmed >= 1
    assert "restarted" in events


async def test_supervisor_gives_up_after_max_restarts():
    engine = FakeEngine(ready=False)  # recovery never succeeds
    events = []

    async def probe():
        return False

    supervisor = EngineSupervisor(
        engine,
        check_interval=0.01,
        max_restarts=2,
        backoff_base=0.01,
        backoff_max=0.01,
        probe=probe,
        on_event=lambda name, data: events.append(name),
    )
    await supervisor.start()
    await asyncio.sleep(0.2)
    await supervisor.stop()

    assert supervisor.gave_up
    assert events.count("gave_up") == 1
