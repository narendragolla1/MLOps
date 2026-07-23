# Conceptual guide

Explanations of OmniAI's building blocks and the reasoning behind them. These pages are about *understanding*; for step-by-step instructions see the [How-to guides](../how_to/index.md).

## Core abstractions

- [Architecture](architecture.md) — how the subsystems fit together and how a message flows through them.
- [Messages](messages.md) — the canonical `OmniMessage` protocol and the chat wire format.
- [Chat models](chat_models.md) — the provider-neutral `ChatModel` interface.
- [Tools](tools.md) — schema generation and validated execution.
- [Agents](agents.md) — the model⇄tools loop.
- [Graphs](graphs.md) — state, merging, compilation, and bounded cycles.

## Infrastructure

- [Serving engines](serving_engines.md) — vLLM/SGLang management, hardware optimization mapping, and the reliability layer.
- [Memory & learning](memory_and_learning.md) — the interaction log and the LoRA adapter lifecycle.
- [Gateway](gateway.md) — channels, adapters, and the interceptor/observer pipeline.
- [Security](security.md) — the threat model and layered defenses.
