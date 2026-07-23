# API reference

Hand-written reference for every public module. Import paths shown are the supported public surface; anything not listed here should be considered internal.

| Module | Contents |
| --- | --- |
| [`omniai.protocol`](protocol.md) | `OmniMessage`, `Role`, `Channel`, `ToolCall` |
| [`omniai.models`](models.md) | `ChatModel`, `ChatResult`, provider adapters, `StructuredOutput` |
| [`omniai.prompts`](prompts.md) | `PromptTemplate`, `ChatPromptTemplate`, placeholders, few-shot |
| [`omniai.graph`](graph.md) | `Graph`, `State`, `tool`, `create_tool_agent` |
| [`omniai.engine`](engine.md) | `ModelEngine`, `EngineConfig`, backends, resilience |
| [`omniai.gateway`](gateway.md) | `GatewayRouter`, channel adapters, security middleware |
| [`omniai.memory`](memory.md) | `InteractionBuffer`, `SkillLoader`, `LoRATrainer`, `ContinuousLearner` |
| [`omniai.guardrails`](guardrails.md) | `PromptGuard`, `GuardrailPolicy` |
| [`omniai.telemetry`](telemetry.md) | `traced_span`, recorder |
| [`omniai.sandbox`](sandbox.md) | `SandboxExecution`, `SandboxResult` |
| [`omniai.evals`](evals.md) | `GoldenDataset`, `AdapterGate`, `EvalVerdict` |
| [`omniai.settings`](settings.md) | `OmniSettings` and every `OMNIAI_*` variable |
