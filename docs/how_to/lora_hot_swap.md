# How to hot-swap LoRA adapters

Load a freshly trained adapter into a **running** server — no restart, no dropped requests.

## One call

```python
await engine.load_lora_adapter("skills-v2", "/adapters/skills-v2")
```

This POSTs to the backend's dynamic-adapter API — `/v1/load_lora_adapter` on vLLM, `/load_lora_adapter` on SGLang — with `{"lora_name": ..., "lora_path": ...}`. By default the new adapter becomes **active**: subsequent `engine.chat(...)` calls route to it by using the adapter name as the model. Pass `activate=False` to load without switching.

## Requirements

- The server must be launched with LoRA enabled — `enable_lora=True` (the default) maps to `--enable-lora` on both backends.
- The adapter path must be visible **to the server process**. In the [Compose stack](deploy_docker_compose.md) the gateway and vLLM containers share an `adapters` volume for exactly this reason.

## Crash safety

The engine records `active_lora` / `active_lora_path`. If the supervised server process crashes, the supervisor restarts it and **re-applies the active adapter** once the server is healthy — a restart never silently reverts you to the base model.

## Automated swapping

The [continuous-learning loop](../tutorials/continuous_learning.md) calls this API for you after each adapter passes the eval gate; you only call it manually for out-of-band adapters.
