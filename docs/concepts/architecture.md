# Architecture

OmniAI is organized as loosely-coupled subsystems that all speak one protocol. Anything can be used standalone (the graph library without the gateway, the engine without the learner), but they compose into a full serving stack.

## The subsystems

| Package | Responsibility |
| --- | --- |
| `omniai.protocol` | The canonical `OmniMessage` every layer exchanges. |
| `omniai.models` | Provider-neutral `ChatModel` (OpenAI, Anthropic, self-hosted engine). |
| `omniai.prompts` | Prompt and chat templates. |
| `omniai.graph` | State graphs, tools, and the prebuilt agent executor. |
| `omniai.engine` | vLLM/SGLang lifecycle, hardware config mapping, LoRA hot-swap, resilience. |
| `omniai.gateway` | FastAPI control plane: channel adapters, security, observability. |
| `omniai.memory` | Skill ingestion, interaction logging, continuous learning. |
| `omniai.guardrails` | Injection screening and PII redaction. |
| `omniai.telemetry` | Tracing with an OTel-optional fallback. |
| `omniai.sandbox` | Isolated Docker execution for generated code. |
| `omniai.evals` | Golden-dataset gating for LoRA adapters. |
| `omniai.settings` / `omniai.app` | Env-driven config and the production app factory. |

## Life of a request

1. A payload arrives on a channel (REST/WS/Discord); the channel **adapter** normalizes it to `OmniMessage`.
2. **Interceptors** run (guardrails: block injection, redact PII). A rejection ends the request with a 400.
3. **Observers** are notified (the interaction buffer logs the inbound message asynchronously).
4. The **handler** runs — typically a compiled graph or agent. Graph nodes call a `ChatModel`; tool calls are validated and executed; the loop continues until a final answer.
5. Model calls reach a provider API or the **ModelEngine**, which fronts the serving backend with retries, a circuit breaker, and a backpressure semaphore.
6. The reply (an `OmniMessage`) passes observers again and is re-encoded by the adapter into the channel's native format.

In the background, the buffer's threshold fires the **ContinuousLearner**: new interactions → training pairs → LoRA adapter → eval gate → hot-swap into the engine.

## Layering rules

- `protocol` has no dependencies; everything depends on it.
- `models` knows nothing about the gateway; graphs know nothing about channels.
- The engine's reliability wrappers (`omniai.engine.resilience`) are reusable primitives — provider adapters use `with_retries` too.
- Production wiring lives in one place: `omniai.app.create_app`, driven entirely by [settings](../reference/settings.md).
