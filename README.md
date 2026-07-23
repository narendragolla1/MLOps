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
| `omniai.engine` | `ModelEngine` factory over vLLM/SGLang subprocesses; maps `quantization="fp8"`, `tensor_parallel_size=2`, etc. to backend CLI flags; OpenAI-compatible async client; dynamic LoRA hot-swap |
| `omniai.graph` | LangGraph-style builder: Pydantic `State`, sync/async nodes, lambda conditional edges, bounded cycles, `@tool` decorator generating JSON Schema from type hints |
| `omniai.gateway` | `GatewayRouter` (FastAPI) with REST, WebSocket, and Discord adapters; interceptor + observer pipeline |
| `omniai.memory` | `skill.md` ingestion into a pre-cached system prompt (RadixAttention-friendly), async SQLite `InteractionBuffer`, background `LoRATrainer` + `ContinuousLearner` cycle |
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
    "backend": "vllm",              # or "sglang"
    "quantization": "fp8",
    "kv_cache": "paged_attention",
    "tensor_parallel_size": 2,
})
await engine.start()                 # launches the server subprocess
text = await engine.chat_text([{"role": "user", "content": "hi"}])
await engine.load_lora_adapter("skills-v2", "/adapters/skills-v2")  # zero downtime
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

## Optional extras

```bash
pip install "omniai[vllm]"       # real vLLM serving
pip install "omniai[sglang]"     # real SGLang serving
pip install "omniai[training]"   # peft/trl LoRA fine-tuning
pip install "omniai[telemetry]"  # OpenTelemetry export
```

The core library, tests, and examples run without GPUs — backend subprocesses are only launched by `engine.start()`, and training falls back to injectable trainers in tests.
