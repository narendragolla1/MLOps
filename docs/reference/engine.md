# omniai.engine

## `EngineConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | Model path/name. |
| `backend` | `"vllm"` | `"vllm"` or `"sglang"`. |
| `host` / `port` | `127.0.0.1` / `8000` | Managed-server bind. |
| `managed` | `True` | `False` = attach to `external_base_url`, spawn nothing. |
| `external_base_url` | `None` | Server URL in unmanaged mode. |
| `quantization` | `None` | e.g. `"fp8"` (vLLM also sets `--kv-cache-dtype fp8`). |
| `kv_cache` | `None` | vLLM: only `paged_attention` valid; SGLang: `"disable_radix"` opts out of RadixAttention. |
| `tensor_parallel_size` | `1` | ≥1. |
| `gpu_memory_utilization` / `max_model_len` | `None` | Mapped per backend. |
| `enable_lora` | `True` | Required for hot-swap. |
| `extra_args` | `{}` | Verbatim CLI flags escape hatch. |
| `request_timeout_s` / `retries` | `120.0` / `3` | HTTP client behavior. |
| `breaker_failure_threshold` / `breaker_reset_s` | `5` / `30.0` | Circuit breaker. |
| `max_concurrent_requests` | `32` | Backpressure semaphore. |

`base_url` property resolves external vs host:port.

## `ModelEngine`

- `ModelEngine.create(config: EngineConfig | dict)` — factory selecting the backend adapter.
- `await start(wait=True, timeout=300.0, supervise=False)` — spawn (managed) or health-check (external); `supervise=True` starts the crash-restart watcher.
- `await stop()` — stop supervision, terminate the subprocess (managed), close the client.
- `set_system_prompt(prompt)` — shared prefix prepended to every conversation.
- `await chat(messages, **kwargs) -> dict` / `chat_text(...) -> str` / `stream_chat(...) -> AsyncIterator[str]`.
- `await load_lora_adapter(name, path, activate=True) -> bool` — dynamic adapter load; active adapter becomes the routed model.
- Attributes: `breaker`, `supervisor`, `active_lora`, `active_lora_path`, `in_flight`, `on_usage` (hook `(prompt_tokens, completion_tokens)`).

## `omniai.engine.resilience`

- `EngineUnavailable` — breaker open / backend down (gateway → 503).
- `await with_retries(fn, attempts=3, base_delay=0.25, max_delay=4.0)` — backoff + full jitter on transient failures (connect errors, 5xx).
- `CircuitBreaker(failure_threshold, reset_timeout)` — `await call(fn)`; states `closed/open/half_open`; single half-open probe.
- `EngineSupervisor(engine, check_interval=2.0, max_restarts=5, ready_timeout=300.0, restart_backoff_base=0.5)` — `start()` / `await stop()`; restarts crashes, re-applies the active LoRA; terminal failure sets `failed` / `failure_reason` (surfaced by readiness).

## Backends

`VLLMAdapter` / `SGLangAdapter` (`omniai.engine.backends`): `build_command()`, `start()`, `stop()`, `await wait_ready(timeout)`, `lora_load_endpoint()`, `lora_load_payload(name, path)`.
