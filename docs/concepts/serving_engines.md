# Serving engines

`ModelEngine` fronts a high-performance serving backend (vLLM or SGLang) with one async, OpenAI-compatible surface — and wraps it in the reliability machinery a production dependency needs.

## Backend abstraction

Users write backend-neutral config (`quantization="fp8"`, `tensor_parallel_size=2`, …); a per-backend adapter maps it to the right CLI flags and knows the backend's health and LoRA endpoints. The mapping table is in the [serving how-to](../how_to/serve_vllm_sglang.md). Two process models:

- **Managed** — the engine spawns the server as a subprocess and owns its lifecycle.
- **External** — the engine attaches to a server run elsewhere (its own container, another host). Same API either way.

## The reliability layer

Every HTTP call to the backend passes through three wrappers (in `omniai.engine.resilience`):

1. **Backpressure semaphore** — at most `max_concurrent_requests` in flight. A traffic burst queues at the gateway instead of piling onto the GPU server, where over-admission degrades everyone's latency.
2. **Retries** — transient failures (connect errors, 5xx) retry with exponential backoff and full jitter. 4xx never retries: that's the caller's bug.
3. **Circuit breaker** — after N consecutive failures the breaker opens and calls fail fast with `EngineUnavailable` (the gateway maps it to 503 + `Retry-After`) instead of stacking timeouts. After a reset window it goes half-open and admits **exactly one probe**; concurrent requests still fail fast, so a recovering server isn't stampeded. Success closes the breaker.

## Supervision

In managed mode, a supervisor task watches the subprocess: on crash it restarts with capped exponential backoff and — critically — **re-applies the active LoRA adapter**, because a fresh server has lost dynamically loaded adapters. If restarts are exhausted, the supervisor records the failure and `/health/ready` reports it; a dead backend is loud, never silent.

## LoRA routing

`load_lora_adapter(name, path)` hits the backend's dynamic-adapter API; the active adapter's name is then used as the `model` field on subsequent chats, which is how vLLM routes requests to it. This is what makes [zero-downtime learning](memory_and_learning.md) possible.

## Prefix caching

`set_system_prompt` maintains a shared conversation prefix. SGLang's RadixAttention (and vLLM's prefix caching) serve that prefix from cache, so multi-kilobyte skill prompts are prefilling work once, not per request.
