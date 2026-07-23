# Introduction

**OmniAI / NexusGraph** is a unified Python framework for **serving** and **orchestrating** Large Language Models. It abstracts hardware-level serving optimizations (vLLM, SGLang) and high-level agentic orchestration (LangGraph-style state graphs) into one cohesive library, with multi-gateway deployments (REST, WebSockets, Discord) and an asynchronous continuous-learning loop that converts interaction history into LoRA adapter updates with zero server downtime.

```mermaid
flowchart LR
    subgraph Gateways
        REST[REST /v1/messages]
        WS[WebSocket /ws]
        DC[Discord webhook]
    end
    REST & WS & DC --> GR[GatewayRouter\nOmniMessage protocol]
    GR --> GD[Guardrails]
    GD --> G[Graph / Agent]
    G --> CM[ChatModel\nOpenAI | Anthropic | Engine]
    CM --> E[ModelEngine\nvLLM | SGLang]
    GR -.observe.-> IB[(InteractionBuffer)]
    IB -->|threshold| CL[ContinuousLearner]
    CL -->|eval gate| EV[AdapterGate]
    EV -->|LoRA hot-swap| E
```

## How to use these docs

| Section | What you'll find |
| --- | --- |
| [Get started](get_started/installation.md) | [Installation](get_started/installation.md) and a [Quickstart](get_started/quickstart.md) that takes you from zero to a tool-calling agent. |
| [Tutorials](tutorials/index.md) | End-to-end walkthroughs: [build an agent](tutorials/build_an_agent.md), [ship a multi-channel chatbot](tutorials/multi_channel_chatbot.md), [set up continuous learning](tutorials/continuous_learning.md). |
| [How-to guides](how_to/index.md) | Short, task-oriented recipes — "How do I do X?" — for everything from [tool calling](how_to/tool_calling.md) to [Docker Compose deployment](how_to/deploy_docker_compose.md). |
| [Conceptual guide](concepts/index.md) | Explanations of the framework's building blocks and the reasoning behind them. |
| [Integrations](integrations/index.md) | Model providers: [OpenAI](integrations/openai.md), [Anthropic](integrations/anthropic.md), and [self-hosted engines](integrations/self_hosted.md). |
| [API reference](reference/index.md) | Hand-written reference for every public module. |
| [Security](security.md) | Security model and how to report vulnerabilities. |
| [Self-hosting deep-dive](../docs/SELF_HOSTING.md) | Design patterns behind the serving layer: LoRA lifecycle, GPU placement, retries/circuit-breaker, supervision. |
| [Compound AI architecture](../docs/COMPOUND_AI.md) | Multi-tenancy, RAG, anti-poisoning curation, anti-forgetting rehearsal, and the shadow-gated learning loop. |

## What can you build?

- **Tool-calling agents** with any provider behind one `ChatModel` interface — see [Build an Agent](tutorials/build_an_agent.md).
- **Multi-channel assistants** served over REST, WebSockets, and Discord from a single graph — see [Multi-channel chatbot](tutorials/multi_channel_chatbot.md).
- **Self-improving deployments** where interaction logs become LoRA adapters, gated by golden-dataset evals and hot-swapped live — see [Continuous learning](tutorials/continuous_learning.md).
- **Multi-tenant business-unit agents** sharing one GPU, each with its own LoRA (behavior) and RAG store (facts), curated and rehearsed against catastrophic forgetting — see [Compound AI architecture](../docs/COMPOUND_AI.md).

## Design principles

1. **One protocol everywhere.** Every payload is normalized to [`OmniMessage`](concepts/messages.md) at the edge; graphs, memory, and providers all speak it.
2. **Providers are interchangeable.** [`ChatModel`](concepts/chat_models.md) hides provider wire formats; agents don't change when the model does.
3. **Production is not an afterthought.** Fail-closed auth, circuit breakers, backpressure, metrics, probes, and migrations ship in the core — see the [security](concepts/security.md) and [serving](concepts/serving_engines.md) concepts.
4. **Test without GPUs.** Every layer has injectable seams (mock transports, fake trainers); the entire test suite runs on a laptop.
