# OmniAI / NexusGraph

A unified, developer-friendly Python framework for **serving** and **orchestrating** Large Language Models. OmniAI abstracts hardware-level serving optimizations (vLLM, SGLang) and high-level agentic orchestration (LangGraph-style graphs) into one cohesive library, with multi-gateway deployments (REST, WebSockets, Discord) and an asynchronous continuous-learning loop that turns interaction history into dynamic LoRA adapter updates — with zero server downtime.

## Architecture

```mermaid
flowchart LR
    subgraph Gateways
        REST[REST /v1/messages]
        WS[WebSocket /ws]
        DC[Discord webhook]
    end
    REST & WS & DC --> GR[GatewayRouter\nOmniMessage protocol]
    GR --> GD[Guardrails\ninjection + PII]
    GD --> G[Graph\nnodes / edges / tools]
    G --> E[ModelEngine\nvLLM | SGLang]
    GR -.observe.-> IB[(InteractionBuffer\nSQLite)]
    IB -->|threshold| CL[ContinuousLearner\nPEFT LoRA]
    CL -->|eval gate| EV[AdapterGate\ngolden dataset]
    EV -->|hot swap| E
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md), organized LangChain-style: [Get started](docs/get_started/installation.md) · [Tutorials](docs/tutorials/index.md) · [How-to guides](docs/how_to/index.md) · [Concepts](docs/concepts/index.md) · [Integrations](docs/integrations/index.md) · [API reference](docs/reference/index.md) · [Security](docs/security.md).

## Quick start

```bash
pip install -e ".[dev]"          # core + test deps
pytest                            # full suite, no GPU required
python examples/basic_agent.py    # full pipeline in < 50 lines
```

## Subsystems

| Module | Purpose |
| --- | --- |
| `omniai.protocol` | Canonical `OmniMessage` flowing through every layer |
| `omniai.engine` | `ModelEngine` factory over vLLM/SGLang subprocesses; maps `quantization="fp8"`, `tensor_parallel_size=2`, `devices=[0,1]`, etc. to backend CLI flags and env; OpenAI-compatible async client with circuit breaker + retries + backpressure; persistent `LoRARegistry` lifecycle (load/unload/rollback/reapply with slot eviction); pluggable backends via `register_backend` — see [docs/SELF_HOSTING.md](docs/SELF_HOSTING.md) |
| `omniai.graph` | LangGraph-style builder: Pydantic `State`, sync/async nodes, lambda conditional edges, bounded cycles, `@tool` decorator generating JSON Schema from type hints |
| `omniai.gateway` | `GatewayRouter` (FastAPI) with REST, WebSocket, and Discord adapters; interceptor + observer pipeline |
| `omniai.rag` | Factual knowledge layer: `VectorStore` contract with dependency-free `InMemoryVectorStore`/`HashEmbedder`, PDF-style chunked ingestion, grounding `Retriever` — facts are retrieved, never trained into weights |
| `omniai.tenancy` | Multi-tenant agents on one base model: `AgentProfile` (LoRA = behavior, retriever = facts), `AgentRegistry`, `TenantHandler` routing each request to its agent's LoRA — see [docs/COMPOUND_AI.md](docs/COMPOUND_AI.md) |
| `omniai.memory` | `skill.md` ingestion into a pre-cached system prompt (RadixAttention-friendly), async `InteractionBuffer` (Postgres/SQLite) with a persisted training watermark, `InteractionJudge` LLM-as-a-judge curation, `RehearsalBuffer` golden-data mixing, background `LoRATrainer` + `ContinuousLearner` cycle with a shadow gate |
| `omniai.guardrails` | `PromptGuard` interceptor: prompt-injection blocking, PII redaction |
| `omniai.telemetry` | OpenTelemetry spans (token counts, latency) with a no-dependency fallback recorder |
| `omniai.sandbox` | `SandboxExecution`: LLM-generated Python/Bash in a locked-down, ephemeral Docker container |
| `omniai.evals` | `AdapterGate`: golden-dataset tool-calling evals that auto-reject regressing LoRA adapters |

## Usage sketches

### Serve a model with hardware optimizations

```python
from omniai.engine import ModelEngine

engine = ModelEngine.create({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",              # or "sglang", or a register_backend() plugin
    "quantization": "fp8",
    "kv_cache": "paged_attention",
    "tensor_parallel_size": 2,
    "devices": [0, 1],              # GPU placement -> CUDA_VISIBLE_DEVICES
    "log_dir": "logs",              # capture server stdout/stderr
})
await engine.start(supervise=True)    # launches the server subprocess + watchdog
text = await engine.chat_text([{"role": "user", "content": "hi"}])
await engine.load_lora_adapter("skills-v2", "/adapters/skills-v2")  # zero downtime
await engine.rollback_lora()          # previous adapter stays loaded — instant revert
```

### Build an agent graph

```python
from omniai.graph import Graph, State, START, END, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather."""
    ...

g = Graph(State)
g.add_node("think", think_fn)
g.add_node("act", act_fn)
g.add_edge(START, "think")
g.add_conditional_edges("think", lambda s: "act" if wants_tool(s) else END)
g.add_edge("act", "think")           # cycles are fine, bounded by max_iterations
app = g.compile()
```

### Continuous learning with an eval gate

```python
from omniai.memory import InteractionBuffer, LoRATrainer, ContinuousLearner
from omniai.evals import AdapterGate, GoldenDataset

buffer = InteractionBuffer("interactions.db", threshold=1000)
gate = AdapterGate(engine, GoldenDataset.from_jsonl("golden.jsonl"))
learner = ContinuousLearner(buffer, LoRATrainer(engine.config.model),
                            engine=engine, evaluator=gate.evaluator)
buffer.on_threshold = learner.trigger   # train + eval + hot-swap at threshold
```

### Multi-tenant agents with RAG, curation, and rehearsal

```python
from omniai.rag import InMemoryVectorStore, Retriever
from omniai.tenancy import AgentProfile, AgentRegistry, TenantHandler
from omniai.memory import InteractionJudge, RehearsalBuffer, ContinuousLearner

store = InMemoryVectorStore()
store.add_texts(["The Widget Pro costs $49 and ships in blue or red."])

registry = AgentRegistry(default="assistant")
registry.register(AgentProfile(
    name="sales", lora="sales-lora-v3",
    system_prompt="You are an enthusiastic sales assistant.",
    retriever=Retriever(store),          # facts, retrieved — never trained into weights
))
handler = TenantHandler(registry, engine)  # routes each request to its agent's LoRA

judge = InteractionJudge(engine=judge_engine, min_score=0.7)   # anti-poisoning
rehearsal = RehearsalBuffer.from_jsonl("golden.jsonl")          # anti-forgetting
learner = ContinuousLearner(
    buffer, LoRATrainer(engine.config.model), engine=engine,
    evaluator=gate.evaluator,   # shadow-gated: scored before it ever goes live
    curator=judge.curate, rehearsal=rehearsal,
)
```

## LangChain-style building blocks

```python
from omniai.models import OpenAIChatModel, AnthropicChatModel, EngineChatModel
from omniai.prompts import ChatPromptTemplate, MessagesPlaceholder
from omniai.graph import create_tool_agent, tool

model = AnthropicChatModel("claude-sonnet-5", api_key=...)   # or OpenAI / self-hosted engine

# Prompt templates with validated variables and history placeholders
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {style} assistant."),
    MessagesPlaceholder("history", optional=True),
    ("user", "{question}"),
])

# Prebuilt tool-calling agent: model <-> tools loop until a final answer
agent = create_tool_agent(model, [get_weather], max_steps=8)
result = await agent.ainvoke({"messages": [OmniMessage(content="weather in Paris?")]})

# Structured output: validated Pydantic objects, retry-on-parse-failure
plan = await model.with_structured_output(TripPlan).invoke("Plan my afternoon")
```

- **`omniai.models`** — one `ChatModel` interface across providers: `OpenAIChatModel` (OpenAI + any OpenAI-compatible endpoint), `AnthropicChatModel` (Messages API with tool_use translation), `EngineChatModel` (self-hosted vLLM/SGLang engine, inheriting its breaker/retries/metrics). Native tool calling on all three.
- **`omniai.prompts`** — `PromptTemplate` / `ChatPromptTemplate` / `MessagesPlaceholder` / few-shot examples.
- **`omniai.graph.create_tool_agent`** — the agent executor loop with schema-validated tool execution; tool errors are fed back to the model instead of crashing the run.
- **`with_structured_output(schema)`** — JSON-schema-guided output with validate-and-retry.

See `examples/tool_agent.py` for a runnable end-to-end demo.

## Production deployment (Docker Compose)

The `deploy/` directory ships a production stack: the **gateway** (this app, non-root image with healthchecks), **Postgres** (interaction log), and **vLLM** (GPU serving container, sharing an `adapters` volume with the gateway so LoRA hot-swaps work by path).

```bash
cp deploy/.env.example deploy/.env   # set OMNIAI_API_KEYS + POSTGRES_PASSWORD
docker compose -f deploy/docker-compose.yml up -d --build
curl -H "X-API-Key: $KEY" localhost:8080/v1/messages -d '{"content":"hi"}'
```

Production behavior out of the box:

- **Fail-closed auth** — the app refuses to boot without `OMNIAI_API_KEYS` (explicit `OMNIAI_AUTH_DISABLED=true` required to run open); per-key token-bucket rate limiting (429 + `Retry-After`), body-size caps, optional CORS.
- **Reliability** — engine calls get retries with exponential backoff + jitter and a circuit breaker (fast 503s with `Retry-After` while the backend is down); managed engines are supervised and restarted with the active LoRA re-applied; graceful shutdown drains and disposes resources.
- **Observability** — Prometheus `/metrics` (request counts/latency, token usage, breaker state, learning cycles), `/health/live` + `/health/ready` probes, structured JSON logs with `X-Request-ID` correlation, optional OTLP trace export (`OMNIAI_OTLP_ENDPOINT`).
- **DB-agnostic persistence** — SQLModel over any SQLAlchemy async URL: Postgres in the compose stack, SQLite for zero-config dev, MySQL etc. unchanged.

All configuration is environment-driven (`OMNIAI_*`, see `deploy/.env.example`); `omniai.app:create_app` is the container entrypoint factory.

## Optional extras

```bash
pip install "omniai[vllm]"       # real vLLM serving
pip install "omniai[sglang]"     # real SGLang serving
pip install "omniai[training]"   # peft/trl LoRA fine-tuning
pip install "omniai[telemetry]"  # OpenTelemetry export
```

The core library, tests, and examples run without GPUs — backend subprocesses are only launched by `engine.start()`, and training falls back to injectable trainers in tests.
