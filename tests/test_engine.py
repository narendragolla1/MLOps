import httpx
import pytest

from omniai.engine import EngineConfig, ModelEngine, SGLangAdapter, VLLMAdapter


def test_factory_selects_backend():
    engine = ModelEngine.create({"model": "meta-llama/Llama-3-8B", "backend": "vllm"})
    assert isinstance(engine.adapter, VLLMAdapter)
    engine = ModelEngine.create({"model": "meta-llama/Llama-3-8B", "backend": "sglang"})
    assert isinstance(engine.adapter, SGLangAdapter)


def test_vllm_command_maps_hardware_optimizations():
    config = EngineConfig(
        model="m",
        backend="vllm",
        quantization="fp8",
        kv_cache="paged_attention",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        extra_args={"enforce_eager": True, "seed": 42},
    )
    cmd = VLLMAdapter(config).build_command()
    assert cmd[:3] == ["vllm", "serve", "m"]
    assert ("--quantization", "fp8") == tuple(cmd[cmd.index("--quantization"):][:2])
    assert "--kv-cache-dtype" in cmd  # fp8 quantization also quantizes the KV cache
    assert ("--tensor-parallel-size", "2") == tuple(cmd[cmd.index("--tensor-parallel-size"):][:2])
    assert ("--gpu-memory-utilization", "0.9") == tuple(
        cmd[cmd.index("--gpu-memory-utilization"):][:2]
    )
    assert ("--max-model-len", "8192") == tuple(cmd[cmd.index("--max-model-len"):][:2])
    assert "--enable-lora" in cmd
    assert "--enforce-eager" in cmd
    assert ("--seed", "42") == tuple(cmd[cmd.index("--seed"):][:2])


def test_vllm_rejects_unsupported_kv_cache():
    config = EngineConfig(model="m", backend="vllm", kv_cache="radix_attention")
    with pytest.raises(ValueError, match="kv_cache"):
        VLLMAdapter(config).build_command()


def test_sglang_command_mapping():
    config = EngineConfig(
        model="m", backend="sglang", quantization="fp8", tensor_parallel_size=4
    )
    cmd = SGLangAdapter(config).build_command()
    assert "sglang.launch_server" in cmd
    assert ("--model-path", "m") == tuple(cmd[cmd.index("--model-path"):][:2])
    assert ("--tp", "4") == tuple(cmd[cmd.index("--tp"):][:2])
    assert "--enable-lora" in cmd


def test_tensor_parallel_must_be_positive():
    with pytest.raises(ValueError):
        EngineConfig(model="m", tensor_parallel_size=0)


def _mock_engine(handler) -> ModelEngine:
    engine = ModelEngine.create({"model": "m", "backend": "vllm"})
    transport = httpx.MockTransport(handler)
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=transport
    )
    return engine


async def test_chat_hits_openai_endpoint_and_injects_system_prompt():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    engine = _mock_engine(handler)
    engine.set_system_prompt("You are helpful.")
    text = await engine.chat_text([{"role": "user", "content": "ping"}])
    assert text == "pong"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert seen["body"]["model"] == "m"


async def test_lora_hot_swap_targets_backend_endpoint_and_activates():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})

    engine = _mock_engine(handler)
    assert await engine.load_lora_adapter("skills-v2", "/adapters/skills-v2")
    assert seen["path"] == "/v1/load_lora_adapter"
    assert seen["body"] == {"lora_name": "skills-v2", "lora_path": "/adapters/skills-v2"}
    assert engine.active_lora == "skills-v2"
