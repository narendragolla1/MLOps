# Product Requirements Document (PRD)
## OmniAI / NexusGraph v1.0

**Document Version:** 1.0  
**Last Updated:** 2026-07-23  
**Status:** In Development  
**Owner:** MLOps Team  

---

## 1. Executive Summary

OmniAI (NexusGraph) is a unified, developer-friendly Python framework designed to abstract the complexity of serving and orchestrating Large Language Models at scale. It bridges the gap between low-level hardware optimization (vLLM, SGLang) and high-level agentic orchestration (LangGraph-style graphs), enabling production-grade LLM applications with minimal operational overhead.

### Key Value Propositions
- **Zero-friction multi-model serving** with hardware-level optimizations transparently applied
- **Unified agent orchestration** via LangGraph-style graphs with deterministic state management
- **Continuous learning capabilities** that turn interaction history into dynamic LoRA adapter updates without service interruption
- **Multi-gateway deployment** (REST, WebSocket, Discord) via a single canonical protocol
- **Production-ready observability** with structured logging, Prometheus metrics, and OpenTelemetry support
- **Secure by default** with fail-closed authentication, prompt injection blocking, and PII redaction

---

## 2. Problem Statement

### Current State
Deploying production LLM applications involves orchestrating multiple specialized systems:
- **Serving frameworks** (vLLM, SGLang) that optimize inference but require manual CUDA tuning and infrastructure plumbing
- **Orchestration libraries** (LangGraph, LangChain) that provide agentic workflows but lack integrated serving
- **Learning systems** (PEFT, TRL) that require offline batch pipelines and manual adapter lifecycle management
- **Gateway layers** (FastAPI, custom REST handlers) that duplicate transport logic across REST/WebSocket/messaging platforms

Teams must integrate these disparate pieces, leading to:
- Steep learning curves for LLM ops expertise
- High operational complexity (manual model loading, adapter management, monitoring)
- Slow iteration on agent capabilities (batch training cycles, manual deployment)
- Fragmented observability across infrastructure and application layers

### Target Users
1. **ML Platform Teams** deploying self-hosted LLM services at scale
2. **Agentic Application Developers** building multi-step LLM-powered workflows
3. **LLM-as-Infrastructure Providers** offering managed agent APIs to customers
4. **Research Teams** experimenting with continuous learning and adaptation

---

## 3. Goals & Success Metrics

### Primary Goals
1. **Reduce time-to-production** for LLM applications from weeks to days
2. **Enable zero-downtime updates** to model serving and adapter configurations
3. **Democratize continuous learning** via automatic interaction-based fine-tuning
4. **Provide unified observability** across serving, orchestration, and learning pipelines

### Success Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| Time to deploy first model | < 5 minutes | Measured from `pip install` to serving HTTP requests |
| Zero-downtime adapter swap | 100ms | LoRA hot-swap without request queueing |
| Learning loop latency | < 1 hour | Collect 1000 interactions → train → eval → deploy |
| First-release adoption | 500+ GitHub stars | Developer community validation |
| Production deployments | 10+ by EOY | Real-world scale validation |
| Test coverage | ≥ 85% | Core modules at ≥ 90% |

---

## 4. Scope & Features

### Phase 1: MVP (v0.1–v1.0)
**Timeline:** Weeks 1–8

#### Core Subsystems
1. **Protocol** (`omniai.protocol`)
   - Canonical `OmniMessage` type flowing through every layer
   - Support for text, tool calls, structured outputs, and streaming
   - Backward-compatible with OpenAI message format
   
2. **Model Engine** (`omniai.engine`)
   - `ModelEngine` factory abstracting vLLM and SGLang backends
   - Automatic CLI flag + environment variable mapping for quantization, tensor parallelism, device placement
   - OpenAI-compatible async client with circuit breaker, retries, and bulkhead isolation
   - LoRA lifecycle management via persistent `LoRARegistry` (load, unload, rollback without restart)
   - Pluggable backend registration for custom serving solutions
   - Supervised process restarts with health checking

3. **Graph Orchestration** (`omniai.graph`)
   - LangGraph-compatible builder API with Pydantic-based state validation
   - Synchronous and asynchronous node support with automatic bridging
   - Lambda conditional edges for state-based routing
   - Bounded cycle support with configurable max iterations
   - `@tool` decorator generating JSON schema from type hints
   - START/END node sentinels

4. **Gateway & Transport** (`omniai.gateway`)
   - FastAPI-based `GatewayRouter` with pluggable adapters
   - REST adapter: OpenAI `/v1/messages`-compatible endpoint
   - WebSocket adapter: streaming responses with bidirectional messaging
   - Discord adapter: webhook ingestion with mention-based routing
   - Interceptor + observer pipeline for auth, rate limiting, request/response transformation
   - Request correlation via X-Request-ID headers

5. **Memory & Continuous Learning** (`omniai.memory`)
   - `skill.md` file ingestion into pre-cached system prompts (RadixAttention-compatible formatting)
   - Async SQLite `InteractionBuffer` for collecting user/assistant exchanges
   - Configurable threshold triggers for training initiation
   - Background `LoRATrainer` integration with PEFT/TRL backends
   - `ContinuousLearner` orchestrator managing train/eval/deploy cycles

6. **Guardrails** (`omniai.guardrails`)
   - `PromptGuard` request interceptor with prompt injection detection
   - PII redaction with configurable patterns and masking strategies
   - Optional output validation against schema constraints
   - Fallback responses on detection (configurable fail-safe)

7. **Observability & Telemetry** (`omniai.telemetry`)
   - OpenTelemetry span generation with token counts and latency
   - Prometheus `/metrics` endpoint (request counters, latency histograms, circuit breaker state)
   - Structured JSON logging with correlation IDs
   - Optional OTLP trace export to external collectors
   - No-dependency fallback recorder for minimal overhead

8. **Sandbox Execution** (`omniai.sandbox`)
   - `SandboxExecution` for safely running LLM-generated Python/Bash scripts
   - Ephemeral Docker container isolation with resource limits
   - Whitelist/blacklist for module imports and system calls
   - Output capture with timeout enforcement

9. **Evals & Quality Gates** (`omniai.evals`)
   - `AdapterGate` for golden-dataset tool-calling evaluations
   - Automatic regression detection and rejection of degraded adapters
   - Metrics tracking (precision, recall, F1) on golden dataset
   - Integration with continuous learning pipeline for automated promotion

#### Model Support
- **Proprietary**: OpenAI (Claude, GPT-4), Anthropic Claude
- **Open source**: Qwen, Llama, Mistral, and any HuggingFace-compatible model via vLLM/SGLang
- **Quantization**: fp8, int8, int4 (configurable per model)
- **Parallelism**: Single-GPU, tensor parallelism across multiple GPUs

#### Deployment Modes
- **Development**: Single-process with SQLite backend
- **Self-hosted**: Docker Compose stack (gateway + vLLM + Postgres)
- **Production**: Multi-replica gateway with load balancing, supervised model servers, persistent interaction storage

### Phase 2: Enhancements (v1.1+)
**Timeline:** Weeks 9+

- Multi-model ensemble inference
- Distributed graph execution across cluster
- Advanced LoRA merging and pruning strategies
- Hugging Face model hub direct integration
- Prefect/Airflow workflow integrations
- Langchain expression language (LCEL) compatibility layer
- Cost optimization and billing integrations

### Out of Scope (v1.0)
- Real-time collaborative multi-user sessions
- Browser-based model fine-tuning UI
- Proprietary quantization schemes
- Kubernetes orchestration (will support in v2.0)
- Multi-tenant RBAC (minimal API key support only in MVP)

---

## 5. Technical Architecture

### System Diagram
```
┌─────────────────────────────────────────────────────────┐
│ Gateways (REST, WebSocket, Discord)                    │
└──────────────────┬──────────────────────────────────────┘
                   │
        OmniMessage Protocol (canonical)
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Gateway Router (FastAPI + Interceptors)                 │
│ ├─ Auth & Rate Limiting                                 │
│ ├─ Request Correlation                                  │
│ └─ Observer Pipeline                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Guardrails                                              │
│ ├─ Prompt Injection Detection                           │
│ └─ PII Redaction                                        │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Graph (LangGraph-style orchestration)                   │
│ ├─ State Management (Pydantic)                          │
│ ├─ Node Execution (Sync/Async)                          │
│ └─ Conditional Routing                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ Model Engine                                            │
│ ├─ vLLM / SGLang Backend Abstraction                    │
│ ├─ Circuit Breaker & Retries                            │
│ ├─ LoRA Registry & Lifecycle                            │
│ └─ Supervised Process Management                        │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
     vLLM / SGLang    Interaction Buffer
     (GPU Serving)    (SQLite/Postgres)
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    Continuous Learner  Golden Dataset    Prometheus
    (Train/Eval/Deploy)  Evaluation      Metrics
```

### Data Model
```
OmniMessage:
  ├─ role: Literal["user", "assistant", "system"]
  ├─ content: str | list[ContentBlock]
  ├─ tool_calls: list[ToolCall] (if assistant-generated)
  ├─ tool_results: list[ToolResult] (if tool-provided)
  └─ metadata: dict (correlation_id, timestamp, etc.)

State (Graph):
  ├─ messages: list[OmniMessage]
  ├─ next_action: str | None (routing hint)
  ├─ iterations: int
  └─ user_context: dict (extensible)

LoRA Adapter:
  ├─ name: str
  ├─ model_id: str
  ├─ path: str (filesystem or URI)
  ├─ created_at: datetime
  ├─ eval_score: float | None
  └─ status: Literal["active", "pending", "rejected"]
```

---

## 6. API Design

### REST Endpoint
```http
POST /v1/messages
Authorization: Bearer <api_key>
X-Request-ID: <correlation_id>

{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"}
  ],
  "max_tokens": 256,
  "stream": false,
  "temperature": 0.7,
  "tools": [{"name": "get_weather", "description": "...", "input_schema": {...}}]
}

Response (200):
{
  "id": "msg_abc123",
  "object": "text_completion",
  "created": 1719100800,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The weather in Paris is 22°C and partly cloudy.",
        "tool_calls": [...]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 18,
    "total_tokens": 60
  }
}
```

### Graph API
```python
from omniai.graph import Graph, State, START, END, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, partly cloudy"

class RouteState(State):
    messages: list[OmniMessage]
    next_action: str | None = None

def route_decision(state: RouteState) -> str:
    last_msg = state.messages[-1].content
    return "weather_tool" if "weather" in last_msg else "respond"

g = Graph(RouteState)
g.add_node("think", think_fn)
g.add_node("weather_tool", weather_node)
g.add_edge(START, "think")
g.add_conditional_edges("think", route_decision, {
    "weather_tool": "weather_tool",
    "respond": END
})
g.add_edge("weather_tool", "think")

app = g.compile()
result = await app.ainvoke({"messages": [...]}, max_iterations=5)
```

### Engine API
```python
from omniai.engine import ModelEngine

engine = ModelEngine.create({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",
    "quantization": "fp8",
    "tensor_parallel_size": 2,
    "devices": [0, 1],
})

await engine.start(supervise=True)
await engine.warmup()

# Chat completion
response = await engine.chat_text([
    {"role": "user", "content": "hello"}
])

# LoRA adapter hot-swap
await engine.load_lora_adapter("skills-v2", "/models/adapters/skills-v2")
await engine.rollback_lora()  # instant revert
```

### Continuous Learning API
```python
from omniai.memory import InteractionBuffer, LoRATrainer, ContinuousLearner
from omniai.evals import AdapterGate, GoldenDataset

buffer = InteractionBuffer("interactions.db", threshold=1000)
gate = AdapterGate(engine, GoldenDataset.from_jsonl("golden.jsonl"))
learner = ContinuousLearner(
    buffer,
    LoRATrainer(engine.config.model),
    engine=engine,
    evaluator=gate.evaluator
)

# Automatically trigger train/eval/deploy at threshold
buffer.on_threshold = learner.trigger
```

---

## 7. Non-Functional Requirements

### Performance
- **First-token latency**: < 500ms for 7B models on single GPU
- **Throughput**: ≥ 200 tokens/sec on A100 with vLLM
- **Adapter swap latency**: < 100ms for in-memory hot-swap
- **Memory overhead**: < 2GB for gateway process (excluding model)

### Reliability
- **Availability**: ≥ 99.5% uptime in production
- **Circuit breaker**: Fail open after 5 consecutive backend errors, retry after 30s
- **Supervised restarts**: Detect crashed vLLM process, restart within 5s, re-apply LoRA
- **Graceful shutdown**: Complete in-flight requests before terminating

### Security
- **Authentication**: API key required for all endpoints (env `OMNIAI_API_KEYS`)
- **Rate limiting**: Token bucket per API key (default: 60 requests/min, 10k tokens/hour)
- **Injection detection**: Block requests matching common SQL/shell injection patterns
- **PII masking**: Redact emails, phone numbers, SSNs from user messages (opt-in)
- **Secrets scanning**: Prevent leaking AWS keys, API tokens in responses (guardrails)

### Observability
- **Logging**: Structured JSON with correlation IDs; aggregate to ELK/Splunk
- **Metrics**: Prometheus `/metrics` scraped every 15s (request rates, latencies, queue depths)
- **Traces**: OpenTelemetry spans for model calls, graph execution, guardrail checks
- **Dashboards**: Grafana templates for request volume, token usage, adapter performance

### Scalability
- **Stateless gateway**: Horizontal scaling via load balancer (Nginx, HAProxy)
- **Database**: PostgreSQL with connection pooling (sqlalchemy.pool.QueuePool)
- **Caching**: In-memory skill prompt caching with LRU eviction
- **Async I/O**: 1000+ concurrent connections per gateway instance

---

## 8. Rollout Plan

### Phase 1: Internal Alpha (Week 1–2)
- Core subsystems feature-complete with unit/integration tests
- Integration test suite covering REST, WebSocket, graph, learning loops
- Documentation: quickstart, tutorial, API reference
- Target: Team dogfooding and feedback

### Phase 2: Closed Beta (Week 3–4)
- Public GitHub repository (open-source MIT license)
- Bug fixes from alpha feedback
- Performance tuning (vLLM subprocesses, LoRA loading)
- Deployment docs for self-hosted stack (Docker Compose)
- Target: 50–100 early adopters

### Phase 3: Public Release (Week 5)
- Stable v1.0.0 release (semver)
- Helm charts for Kubernetes (optional extras)
- Production deployment guide with monitoring templates
- Launch blog post + HN/Twitter announcement
- Target: Public availability and adoption

### Phase 4: Production Support (Week 6+)
- Monitor production deployments for issues
- Weekly release cadence for bug fixes
- Quarterly minor releases (v1.1, v1.2) for feature additions
- Roadmap transparency via GitHub discussions

---

## 9. Success Criteria

### Functional
- [ ] All subsystems pass integration tests (≥ 85% coverage)
- [ ] REST `/v1/messages` endpoint OpenAI-compatible
- [ ] LoRA hot-swap works with zero-downtime for active requests
- [ ] Continuous learning loop completes (collect → train → eval → deploy) in < 1 hour for 1000 interactions

### Non-Functional
- [ ] First-token latency ≤ 500ms on 7B model (vLLM backend)
- [ ] Adapter swap ≤ 100ms
- [ ] Circuit breaker activates within 5 retries
- [ ] Graceful shutdown completes in < 10s

### Adoption
- [ ] 500+ GitHub stars by end of Phase 3
- [ ] 10+ production deployments by end of Q3
- [ ] Positive community sentiment (NPS ≥ 40)

---

## 10. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| vLLM API instability | High | Vendor-neutral backend abstraction; support SGLang alternative |
| Adapter regression in production | High | Mandatory golden-dataset eval before deployment; version pinning |
| Memory leaks in long-running gateway | High | Periodic profiling; event loop monitoring; max worker age |
| Database scaling bottleneck | Medium | Connection pooling; read replicas for queries; async migrations |
| Slow adoption due to learning curve | Medium | Comprehensive tutorials; example scripts; Slack community |

---

## 11. Open Questions & Future Considerations

1. **Multi-tenancy**: Should v1.1 support RBAC for shared deployments?
2. **Distributed graphs**: Execution across multiple machines (v2.0)?
3. **Cost optimization**: Integration with spot instances / cheaper GPUs?
4. **Langchain parity**: Full expression language (LCEL) compatibility?
5. **Quantization expansion**: Support for custom GGUF / AWQ formats?

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **OmniMessage** | Canonical message format flowing through all layers (role, content, tool_calls, etc.) |
| **LoRA** | Low-Rank Adaptation; parameter-efficient fine-tuning technique for adapters |
| **Circuit breaker** | Fault-tolerance pattern; fails fast during cascading errors |
| **vLLM** | Inference framework with paged attention and continuous batching |
| **SGLang** | Structured generation framework optimizing constrained decoding |
| **Adapter** | Lightweight model weights (~100MB) loaded on top of a base model |
| **Golden dataset** | Curated QA pairs for evaluating adapter quality |
| **Guardrails** | Security layer detecting injection, PII, and enforcing constraints |
| **Telemet** | Observability subsystem (logging, metrics, traces) |

---

## Appendix B: References
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/hpcaitech/sglang
- LangGraph: https://github.com/langchain-ai/langgraph
- PEFT: https://github.com/huggingface/peft
- Prometheus: https://prometheus.io/
- OpenTelemetry: https://opentelemetry.io/
