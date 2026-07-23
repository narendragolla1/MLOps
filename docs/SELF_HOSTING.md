# Self-Hosting LLMs with OmniAI: Design Patterns & Practices

This guide explains how OmniAI's serving layer is put together, which design
patterns each piece uses and why, the common failure modes of self-hosted LLM
serving and the technique the framework applies (or recommends) for each. The
goal of the API is that all of this stays behind one simple entry point:

```python
from omniai.engine import ModelEngine

engine = ModelEngine.create({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",              # or "sglang", or your own registered backend
    "quantization": "fp8",
    "tensor_parallel_size": 2,
    "devices": [0, 1],              # -> CUDA_VISIBLE_DEVICES for this server only
    "max_loras": 8,                 # LoRA slots; oldest evicted automatically
    "log_dir": "logs",              # backend stdout/stderr captured here
    "max_concurrent_requests": 64,  # bulkhead: queue here, not in timeouts
})
await engine.start(supervise=True)   # launches the server + a restart watchdog
await engine.warmup()
```

Attaching to an already-running server (e.g. the vLLM container in the
compose stack) instead of spawning one is one flag away: `managed=False,
external_base_url="http://vllm:8000"`.

---

## 1. The pattern map

Every subsystem in the serving layer is one classic pattern, applied narrowly:

| Piece | Pattern | Why this pattern |
| --- | --- | --- |
| `ModelEngine` | **Facade** | One object owns lifecycle, chat, LoRA, health; callers never see backend detail. |
| `ModelEngine.create(...)` | **Factory Method** | Config (`backend: "vllm"`) selects the concrete adapter; adding a backend never changes calling code. |
| `BackendAdapter` + `VLLMAdapter`/`SGLangAdapter` | **Adapter** | Each backend's CLI flags, env vars, and LoRA endpoints differ; the adapter normalizes them to one interface. |
| `BackendAdapter.build_env()` / `start()` / `stop()` | **Template Method** | The base class owns the invariant plumbing (process groups, log capture, GPU placement); subclasses fill in only the hooks (`_backend_env`, `build_command`, `lora_*_endpoint`). |
| `register_backend(name, cls)` | **Registry (open/closed)** | Third-party backends (TGI, llama.cpp server, Triton) plug in without editing the framework. |
| `with_retries` + `CircuitBreaker` | **Strategy + Circuit Breaker** | Transient failures get exponential backoff + jitter; sustained failure trips a three-state breaker (closed → open → half-open) so a dead backend fails fast instead of piling up retries. |
| `max_concurrent_requests` semaphore | **Bulkhead** | A saturated backend queues at the engine boundary instead of exploding into timeouts everywhere. |
| `LoRARegistry` | **Registry + Memento** | The server only knows load/unload; the registry remembers *which adapter is production*, its predecessor (rollback), and survives restarts via JSON persistence. |
| `EngineSupervisor` | **Watchdog / Supervisor** with **crash-loop breaker** | GPU servers OOM and segfault routinely; recovery (restart → readiness → re-apply LoRA) must be automatic, and give-up-after-N restarts prevents thrashing a broken GPU. |
| `managed` / `external_base_url` | **Strategy (attach vs. own)** | The same engine API drives a subprocess it owns or health-checks a server owned by something else (a compose stack, a Kubernetes deployment). |
| `extra_args` / `env` passthrough | **Escape hatch** | Abstractions over fast-moving servers must never trap the user; any flag/env the backend grows tomorrow is reachable today. |

The composition rule: **Facade in front, Adapters below, resilience wrapped
around every call, Registries for anything with a lifecycle, a Supervisor
around anything with a process.** That combination is what lets one config
dict express `{backend} × {quantization} × {parallelism} × {placement} ×
{LoRA}` without special cases.

## 2. How the abstract methods compose

`BackendAdapter` is deliberately small. To support a new backend you implement
exactly six hooks — everything else (process groups, log files, health
polling, eviction, retries, supervision) is inherited:

```python
class TGIAdapter(BackendAdapter):
    def build_command(self) -> list[str]: ...          # CLI launch line
    def _backend_env(self) -> dict[str, str]: ...      # optional: env vars
    def lora_load_endpoint(self) -> str: ...
    def lora_load_payload(self, name, path) -> dict: ...
    def lora_unload_endpoint(self) -> str: ...
    def lora_unload_payload(self, name) -> dict: ...

register_backend("tgi", TGIAdapter)
engine = ModelEngine.create({"model": "m", "backend": "tgi"})
```

Because the hooks are orthogonal, every combination in the config matrix works
on every backend that supports it — and a backend that *doesn't* support a
combination should raise in `build_command` (see `VLLMAdapter`'s `kv_cache`
rejection) rather than silently ignore it. **Fail loudly at build time, never
quietly at serve time.**

## 3. Common self-hosting problems → technique used

| Problem you will hit | Technique | Where |
| --- | --- | --- |
| Backend crashes at boot with no clue why | Capture stdout/stderr to `log_dir` instead of discarding | `BackendAdapter._open_log` |
| Dynamic LoRA API rejected by vLLM | Server env is a first-class hook; vLLM adapter sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `VLLMAdapter._backend_env` |
| Two engines fight over GPU 0 | Declarative placement: `devices=[2,3]` → `CUDA_VISIBLE_DEVICES`, validated against `tensor_parallel_size` | `EngineConfig.devices` |
| Orphaned tensor-parallel workers keep GPU memory after a kill | Process-group lifecycle: `start_new_session=True` + `killpg`, TERM then KILL | `BackendAdapter.start/stop` |
| First request after boot is slow (CUDA graph capture) | Explicit warm-up generation after readiness | `ModelEngine.warmup` |
| Transient 503/timeout during load spikes | Exponential backoff + jitter, only on retryable statuses | `with_retries` |
| Backend is down for an extended outage | Circuit breaker fails fast (`EngineUnavailable`, mapped to HTTP 503 + `Retry-After`) instead of retrying into a dead server | `CircuitBreaker` |
| One slow backend melts every caller | Bulkhead admission at the engine | `max_concurrent_requests` |
| Skill prompts re-prefilled on every request | Prefix caching on by default on both backends | `prefix_caching` flag mapping |
| Continuous learning fills all LoRA slots and stalls | Slot-aware eviction: oldest non-active, non-rollback adapter unloaded automatically | `ModelEngine.load_lora_adapter` |
| Bad adapter went live, need out *now* | `rollback_lora()` — predecessor stays loaded, rollback is one cheap call | `LoRARegistry.previous` |
| Server restarted, forgot its adapters | Persistent registry + `reapply_active_lora()` on recovery | `LoRARegistry(persist_path=...)`, `EngineSupervisor` |
| Server dies at 3 a.m. | Supervisor: health watchdog, restart with backoff, re-applies the active LoRA, gives up (and reports why) after `max_restarts` | `EngineSupervisor` |
| Health endpoint says OK but process is dead | Health = process liveness **and** HTTP probe | `ModelEngine.health` |

## 4. Techniques to reach for as requirements grow

Not everything belongs in the framework. Guidance for the next tier of
requirements, in the order they usually arrive:

- **Throughput tuning.** Start with the backend's own knobs via `extra_args`
  (`max_num_seqs`, `enforce_eager=False`, chunked prefill). Scrape the
  backend's `/metrics` (KV-cache utilization, queue depth) and feed it into
  admission decisions before buying more GPUs.
- **Quantization choice.** fp8 for Hopper+ GPUs (couples KV-cache savings),
  AWQ/GPTQ **require pre-quantized checkpoints** — pick the checkpoint first,
  then the flag. Validate accuracy with your eval gate, not vibes.
- **Multi-replica serving.** The engine is single-server by design; scale out
  by running N engines behind a load balancer and moving the `LoRARegistry`'s
  persistence to shared storage (DB) so "the production adapter" is a
  server-side alias all replicas resolve — activation becomes an atomic
  registry write, not N client-side flips.
- **Training vs serving.** Never co-locate by accident: give training its own
  `devices` (or node) and treat it as a queued job. The serving GPU's memory
  is already spoken for (`gpu_memory_utilization`).
- **Artifact distribution.** Local adapter paths only work on one box. Push
  adapters to object storage and stage them onto serving hosts; keep the
  registry as the source of truth for name → version → artifact → eval verdict.
- **Context management.** Enforce a tokenizer-aware budget before the server
  does it for you with a 400: truncate or summarize history to
  `max_model_len` minus generation headroom.
- **Streaming.** Expose `stream_chat` end-to-end (SSE/WebSocket) — time to
  first token is the metric users feel, and self-hosting gives you full
  control over it.

## 5. Practices the framework enforces by default

1. **Own the process tree** — every server runs in its own process group with
   captured logs; `stop()` provably releases the GPU.
2. **Environment is configuration** — placement and backend gates are config
   fields, not shell exports someone forgets.
3. **Every mutation has an inverse** — load/unload, activate/rollback,
   start/stop, managed/attach. If you can't undo it, you can't operate it.
4. **State that outlives the process is persisted** — the adapter registry
   survives restarts; in-memory-only state is treated as cache.
5. **Fail fast on invalid combinations** — unsupported `kv_cache`, devices
   fewer than `tensor_parallel_size`, unknown backends: all rejected at
   config/build time.
6. **Escape hatches everywhere** — `extra_args`, `env`, `register_backend`.
   The abstraction is a default, never a cage.
