"""Self-hosting features: env plumbing, LoRA lifecycle, custom backends.

Client-side retries/circuit-breaker/supervision are already covered end to
end by test_resilience.py against omniai.engine.resilience — this file only
covers what's new here: GPU placement/env building, prefix-caching and
LoRA-slot flag mapping, pluggable backend registration, and the
LoRARegistry-backed adapter lifecycle (eviction, rollback, persistence).
"""

import httpx
import pytest

from omniai.engine import (
    ADAPTERS,
    EngineConfig,
    LoRARegistry,
    ModelEngine,
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


def test_no_device_constraint_when_devices_unset():
    # devices=None means "let the scheduler decide" -> no validation error.
    EngineConfig(model="m", tensor_parallel_size=4)


# -- command mapping ----------------------------------------------------------


def test_vllm_prefix_caching_and_lora_slots():
    cmd = VLLMAdapter(EngineConfig(model="m", backend="vllm", max_loras=8)).build_command()
    assert "--enable-prefix-caching" in cmd
    assert tuple(cmd[cmd.index("--max-loras") :][:2]) == ("--max-loras", "8")
    cmd = VLLMAdapter(EngineConfig(model="m", backend="vllm", prefix_caching=False)).build_command()
    assert "--enable-prefix-caching" not in cmd


def test_sglang_prefix_caching_opt_out_and_lora_slots():
    cmd = SGLangAdapter(EngineConfig(model="m", backend="sglang", max_loras=2)).build_command()
    assert "--disable-radix-cache" not in cmd
    assert tuple(cmd[cmd.index("--max-loras-per-batch") :][:2]) == ("--max-loras-per-batch", "2")
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


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        ModelEngine.create({"model": "m", "backend": "not-a-backend"})


# -- process lifecycle ---------------------------------------------------------


def test_adapter_not_alive_before_start():
    adapter = VLLMAdapter(EngineConfig(model="m", backend="vllm"))
    assert adapter.is_alive() is False


def test_log_dir_captures_backend_output(tmp_path):
    config = EngineConfig(model="m", backend="vllm", log_dir=str(tmp_path))
    adapter = VLLMAdapter(config)
    # Swap the real launch command for a harmless one so the test doesn't
    # depend on vllm being installed; start() still exercises log plumbing.
    adapter.build_command = lambda: ["python3", "-c", "print('hi')"]
    adapter.start()
    adapter.process.wait(timeout=10)
    log_file = tmp_path / f"vllm-{config.port}.log"
    assert log_file.exists()
    assert b"hi" in log_file.read_bytes()
    adapter.stop()


# -- LoRA lifecycle -------------------------------------------------------------


def _engine_with_handler(handler, **config) -> ModelEngine:
    engine = ModelEngine.create({"model": "m", "backend": "vllm", **config})
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    return engine


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
    assert engine.active_lora_path == "/a/v1"
    # Roll forward again — v2 stayed loaded.
    assert await engine.rollback_lora() == "v2"


async def test_rollback_with_no_history_deactivates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    engine = _engine_with_handler(handler)
    await engine.load_lora_adapter("v1", "/a/v1")
    assert await engine.rollback_lora() is None
    assert engine.active_lora is None


async def test_reapply_active_lora_after_restart():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    engine = _engine_with_handler(handler)
    await engine.load_lora_adapter("v1", "/a/v1")
    assert await engine.reapply_active_lora()
    assert requests.count("/v1/load_lora_adapter") == 2


async def test_reapply_with_nothing_active_is_noop():
    engine = _engine_with_handler(lambda r: httpx.Response(200, json={"status": "ok"}))
    assert await engine.reapply_active_lora() is False


async def test_health_reports_process_and_server_state():
    engine = _engine_with_handler(lambda r: httpx.Response(200))
    status = await engine.health()
    assert status["process"] is False  # never started in this test
    assert status["server"] is True
    assert status["active_lora"] is None


async def test_warmup_returns_false_on_failure():
    engine = _engine_with_handler(lambda r: httpx.Response(500))
    assert await engine.warmup() is False


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


def test_activate_unknown_adapter_raises():
    with pytest.raises(KeyError):
        LoRARegistry().activate("nope")


def test_eviction_candidate_none_below_capacity():
    registry = LoRARegistry()
    registry.register("v1", "/a/v1")
    assert registry.eviction_candidate(capacity=4) is None
