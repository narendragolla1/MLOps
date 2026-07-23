# How to serve models with vLLM or SGLang

## Managed mode: the engine owns the server process

```python
from omniai.engine import ModelEngine

engine = ModelEngine.create({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",               # or "sglang"
    "quantization": "fp8",
    "kv_cache": "paged_attention",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
})
await engine.start(supervise=True)   # spawns the server, waits for /health, watches it
text = await engine.chat_text([{"role": "user", "content": "hi"}])
await engine.stop()
```

The config is backend-neutral; each adapter maps it to the right CLI flags:

| Config | vLLM | SGLang |
| --- | --- | --- |
| `quantization="fp8"` | `--quantization fp8 --kv-cache-dtype fp8` | `--quantization fp8` |
| `kv_cache` | PagedAttention is native (no flag); other values rejected | RadixAttention is default; `"disable_radix"` opts out |
| `tensor_parallel_size=2` | `--tensor-parallel-size 2` | `--tp 2` |
| `gpu_memory_utilization=0.9` | `--gpu-memory-utilization 0.9` | `--mem-fraction-static 0.9` |
| `max_model_len=8192` | `--max-model-len 8192` | `--context-length 8192` |
| `enable_lora=True` (default) | `--enable-lora --max-loras 4` | `--enable-lora` |
| `extra_args={"seed": 42}` | `--seed 42` (verbatim escape hatch) | same |

`supervise=True` restarts a crashed server with capped backoff and re-applies the active LoRA adapter; terminal failures surface in `/health/ready` (see [observability](observability.md)).

## External mode: attach to a server you run elsewhere

In the [Compose stack](deploy_docker_compose.md) vLLM is its own container — the engine attaches instead of spawning:

```python
engine = ModelEngine.create({
    "model": "served-model",
    "managed": False,
    "external_base_url": "http://vllm:8000",
})
await engine.start()      # health-checks only; no subprocess
```

Set via env: `OMNIAI_ENGINE_BASE_URL` + `OMNIAI_ENGINE_MANAGED=false`.

## Reliability knobs

`request_timeout_s`, `retries`, `breaker_failure_threshold`, `breaker_reset_s`, and `max_concurrent_requests` (backpressure) are all `EngineConfig` fields — details in [serving engine concepts](../concepts/serving_engines.md).

## Pre-cached system prompts

`engine.set_system_prompt(...)` prepends a shared prefix to every conversation — combined with `SkillLoader` (skill.md ingestion) this exploits SGLang's RadixAttention / vLLM prefix caching so long capability prompts cost prefill once.
