# Self-hosted (vLLM / SGLang)

`EngineChatModel` puts your own GPU deployment behind the same `ChatModel` interface as the cloud providers — while keeping everything `ModelEngine` provides: process supervision, circuit breaker, retries, backpressure, token metrics, and LoRA adapter routing.

```python
from omniai.engine import ModelEngine
from omniai.models import EngineChatModel

engine = ModelEngine.create({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",              # or "sglang"
    "quantization": "fp8",
    "tensor_parallel_size": 2,
})
await engine.start(supervise=True)

model = EngineChatModel(engine)     # drop-in wherever a ChatModel goes
agent = create_tool_agent(model, tools)
```

## Why not just `OpenAIChatModel(base_url=...)`?

That works for plain inference against an already-running server. `EngineChatModel` is the right choice when you want the engine's extra machinery:

- requests automatically route to the **active LoRA adapter** after a [hot-swap](../how_to/lora_hot_swap.md);
- failures trip the **circuit breaker** the gateway's 503s and readiness probe key off;
- a **backpressure semaphore** protects the GPU server from bursts;
- token usage feeds the Prometheus metrics;
- in managed mode, the **supervisor** restarts crashes and re-applies adapters.

## Deployment shapes

- **Same process (managed)**: the engine spawns and supervises the server — simplest for a single box. See [serving how-to](../how_to/serve_vllm_sglang.md).
- **Separate container (external)**: `managed=False` + `external_base_url` — the shape the [Compose stack](../how_to/deploy_docker_compose.md) uses, with the official `vllm/vllm-openai` image.

Hardware optimization mapping (fp8, KV cache, tensor parallelism, memory fraction) is covered in the [serving engines concept](../concepts/serving_engines.md).
