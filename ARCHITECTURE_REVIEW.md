# OmniAI / NexusGraph — 360° Architecture Review

**Scope:** every module in `omniai/` (~1,600 LOC), `examples/basic_agent.py`, the test suite (68 tests, all passing), and packaging.
**Verdict in one line:** the architecture is genuinely well-layered with excellent testability seams, but the flagship feature — the continuous-learning pipeline — is broken by an ordering bug and silently swallows its own failures, the gateway is completely unauthenticated, and the system as designed cannot run more than one replica. It is a strong prototype, not a production system.

---

## 1. Critical bugs (correctness)

### 1.1 The eval gate evaluates the adapter *before* it is loaded — the flagship pipeline cannot succeed

`ContinuousLearner.run_cycle` (`omniai/memory/learning.py:188-205`) runs the evaluator **before** calling `engine.load_lora_adapter(name, path)`:

```python
name, path = await self.trainer.train(pairs)
if self.evaluator is not None:
    verdict = self.evaluator(name, path)      # <-- adapter not on the server yet
    ...
if self.engine is not None:
    await self.engine.load_lora_adapter(name, path)
```

`AdapterGate.score` (`omniai/evals/gate.py:97-111`) evaluates by sending `model=<adapter_name>` to the live server. Since the adapter has not been loaded, every request returns 4xx, `resp.raise_for_status()` (`omniai/engine/engine.py:91`) raises, and the whole cycle aborts. The advertised train → gate → hot-swap loop can never complete against a real backend. The gate needs the adapter loaded (but not activated) before scoring, and unloaded again on rejection.

### 1.2 The learning cycle's failures are invisible

`ContinuousLearner.trigger` (`learning.py:212-214`) creates a fire-and-forget task:

```python
def trigger(self):
    return asyncio.get_running_loop().create_task(self.run_cycle())
```

- No reference is retained by the class, so the task can be garbage-collected mid-flight (documented asyncio footgun).
- No `add_done_callback` / exception handling: any failure in train/eval/swap (including bug 1.1) disappears without a log line. Combined with the fact that the codebase contains **zero uses of the `logging` module**, a production operator has no way to know training has been failing for weeks.

### 1.3 Sandbox timeout leaks the running container

`SandboxExecution._run_subprocess` (`omniai/sandbox/executor.py:79-83`) on timeout does:

```python
proc.kill()   # kills the *docker CLI client*, not the container
```

`docker run` is a client attached to the daemon; killing it detaches but the container **keeps executing untrusted LLM-generated code** with no wall-clock cap (only `--rm` on eventual exit). An attacker who induces an infinite loop accumulates runaway containers. Fix: run with `--name` + `docker kill <name>` on timeout, or wrap the in-container command with `timeout(1)`.

Additional sandbox issues:
- Code is passed as an argv element (`executor.py:69`): visible to every process on the host via `ps`, and breaks at ARG_MAX for large payloads. Pipe via stdin instead.
- No `--pids-limit` → fork bombs are uncontained within the memory cap.
- `stdout=PIPE` with no size cap: sandboxed code that prints gigabytes OOMs the *host* process reading it.
- No graceful handling when Docker is absent — `FileNotFoundError` propagates raw.

### 1.4 Every cycle retrains on the entire history

`run_cycle` calls `self.buffer.fetch()` with no watermark (`learning.py:180`), so cycle *N* re-trains on all data from cycles 1..N-1 plus the new slice: compounding cost, repeated exposure of old data (catastrophic-forgetting dynamics inverted), and adapters that drift toward the oldest data distribution. A `last_trained_id`/timestamp cursor is required.

Related: `LoRATrainer._version` (`learning.py:108,114`) is in-memory. After a restart it resets to 0 → the next adapter is again `...-lora-v1`, colliding with the existing output directory and the name already registered in the serving backend.

### 1.5 Baseline drift in the eval gate

`AdapterGate.establish_baseline` (`gate.py:113-116`) calls `score()` with no model override, which resolves to `self.active_lora or self.config.model` (`engine.py:83`). After the first successful swap, a lazily-established baseline is measured against the *current adapter*, not the base model. Each generation is then only required to beat its parent, allowing monotonic degradation across generations (ratchet effect).

Also: `score()` calls `chat_text` without pinning `temperature=0`/seed — the gate is non-deterministic, so accept/reject decisions are partly noise.

### 1.6 `CompiledGraph.invoke` breaks inside any event loop

`graph.py:137-141` uses `asyncio.run()`. Called from any async context (which is everywhere in this framework — gateway handlers, nodes, tests with `asyncio_mode=auto`), it raises `RuntimeError: asyncio.run() cannot be called from a running event loop`.

### 1.7 Observers are not "fire-and-forget" and blocked messages are never audited

The docstring promises fire-and-forget observers (`omniai/gateway/router.py:45-47`), but `dispatch` awaits every observer serially **in the request path** (`router.py:87-92`):

- A slow SQLite write adds latency to every user request; an observer exception turns into a 500 for the user.
- Interceptors run *before* the inbound `_notify`, so a message blocked by `PromptGuard` is never logged — **there is no audit trail of attacks**, which is precisely the traffic you want recorded.
- PII redaction also runs before logging, so the buffer stores redacted text — good for privacy, but it means the training corpus is salted with literal `[REDACTED:email]` tokens the model will learn to emit.

### 1.8 The protocol silently drops tool semantics

`OmniMessage.to_openai()` (`omniai/protocol.py:62-64`) returns only `{role, content}`:
- `tool_calls` are dropped for assistant messages;
- `tool` role messages carry no `tool_call_id`.

Any real OpenAI-compatible backend will reject or mis-handle such histories. The protocol is therefore not actually round-trippable through the very API contract the engine targets.

### 1.9 System-prompt injection is silently skipped

`ModelEngine._build_messages` (`engine.py:71-74`) prepends the cached skill prompt only when the caller's first message is not a `system` message. A caller that passes its own system message silently loses **all installed skills** — no merge, no warning.

### 1.10 Threshold race and O(n) counting in the buffer

`InteractionBuffer.log` (`omniai/memory/buffer.py:96-106`):
- `COUNT(*)` on every single message — O(n) per request, forever growing.
- The count check and `_threshold_fired_at` update are not atomic across concurrent `log` calls on the same loop tick boundary (two awaits interleave), so the trigger can double-fire or skip.
- `_threshold_fired_at` is set *before* the cycle succeeds; a failed cycle (bug 1.1/1.2) consumes the threshold and the next attempt is another `threshold` messages away.
- `INSERT OR REPLACE` (`buffer.py:63`) silently destroys an existing row on id collision instead of surfacing the anomaly.

---

## 2. Security review

### 2.1 No authentication, anywhere

Every endpoint — `POST /v1/messages`, `POST /discord/webhook`, `WS /ws` (`router.py:101-133`) — is anonymous. The example binds `0.0.0.0` (`examples/basic_agent.py:52`). Anyone with network reach gets free LLM inference, can poison the training buffer (see 2.4), and can pump the interaction count to force LoRA training cycles (resource-exhaustion via GPU training). There is also no rate limiting and no request-size cap.

### 2.2 Discord webhook is spoofable and protocol-incomplete

Discord requires Ed25519 verification of `X-Signature-Ed25519`/`X-Signature-Timestamp` on interaction endpoints and a type-1 PING→PONG handshake; endpoints that skip verification are rejected by Discord and, worse, accept forged payloads from anyone. `DiscordAdapter` (`omniai/gateway/adapters.py:59-81`) implements neither, and its reply shape `{"content": ...}` is not a valid interaction response (`{"type": 4, "data": {...}}`).

### 2.3 Guardrails are regex theater

`PromptGuard` (`omniai/guardrails/middleware.py`):
- Injection detection is five English regexes. Base64, leetspeak, any non-English language, or trivial rephrasing ("pay no attention to earlier guidance") sail through. Regex screening is a fine *first* layer but is presented as *the* layer; there is no model-based classifier hook and no guard on the **output** direction (system-prompt leakage, PII generated by the model).
- PII over-matching: the `ipv4` pattern matches version strings ("upgrade to 10.2.3.4"), `credit_card` matches any 13-16 digit run (order IDs, tracking numbers), corrupting legitimate content — which then flows into training data (1.7).
- Redaction is irreversible with no vault/tokenization option, and blocked messages are unlogged (1.7).

### 2.4 Training on unfiltered production traffic is a poisoning vector

The buffer logs *everything* and the learner trains on *everything* (`learning.py:24-55`): no quality signal (thumbs-up, task success), no dedup, no adversarial filtering, and — critically — the model's **own outputs** become its next training set (self-consumption loop → degeneration). An unauthenticated attacker (2.1) can inject arbitrary pairs and steer the model; the only backstop is the eval gate, which is currently broken (1.1) and only measures tool-calling accuracy, not safety or content regressions.

Compliance angle: user conversations are retained indefinitely (no TTL/pruning) and used for training with no consent flag, no per-session opt-out, and no erasure path (GDPR Art. 17 problem).

### 2.5 WebSocket robustness

`websocket_endpoint` (`router.py:119-133`) catches only `WebSocketDisconnect` and `GuardrailViolation`. Malformed JSON in `receive_json` or any handler exception kills the connection uncleanly; there is no per-connection auth, no idle timeout, no message-size limit, and no backpressure.

---

## 3. Reliability & robustness

| Gap | Evidence | Consequence |
| --- | --- | --- |
| No retries/backoff/circuit breaker on engine calls | `engine.py:90-92` — single `post` + `raise_for_status` | One transient backend hiccup = user-facing 500; a down backend = every request 500s with no fast-fail |
| Backend subprocess logs discarded | `backends.py:53-57` — `stdout/stderr → DEVNULL` | Crash-on-boot (OOM, bad flag, missing model) is undiagnosable; `wait_ready` just times out after 300 s with no cause |
| No process supervision | `backends.py` has no liveness check after start | If vLLM crashes mid-run, the engine keeps posting to a dead port until humans notice |
| `/health` lies | `router.py:97-99` returns `{"ok"}` unconditionally | Orchestrators (k8s) keep routing traffic to a gateway whose engine is dead |
| Zero logging | `grep -r "import logging" omniai/` → nothing | No operational visibility at all; telemetry fallback `recorder` (`telemetry/__init__.py:53`) is an **unbounded in-memory list** — a slow memory leak in any long-lived process that lacks the otel SDK |
| No graceful shutdown | no lifespan hooks; engine subprocess orphaned if uvicorn dies uncleanly | GPU stays occupied by an orphan server |
| Config only via code | `EngineConfig` has no env/file loading | Secrets and ports end up hardcoded; no 12-factor deploys |
| No CI, no lint/type-check config | repo has no `.github/workflows`, no ruff/mypy config | Quality relies entirely on local discipline |

Smaller correctness hazards:
- `State.merge` (`omniai/graph/state.py:28-42`) does a full `model_dump()` + re-validate of the entire state per node step — O(history) per step, O(n²) per conversation — and makes it *impossible for a node to replace/truncate `messages`* (append-only is hardwired with no `Annotated` reducer mechanism à la LangGraph).
- `Graph` allows nodes with no outgoing edge to silently terminate (`graph.py:117` defaults to `END`) — a misspelled `add_edge` source is a silent no-op rather than a compile error; `compile()` validates edge *targets* but not that every node is reachable or has an exit.
- `SkillLoader.load_file` appends on every call (`skills.py:69`) — re-loading a directory duplicates every skill in the composed prompt; the "YAML-ish" frontmatter parser (`skills.py:39-50`) breaks on real YAML (lists, multiline strings, quoted colons).
- `stream_chat` (`engine.py:117-129`) does `import json` inside the async loop, assumes `choices[0]` exists on every chunk, and has no handling for backend SSE error events.
- Gateway request bodies are `dict[str, Any]` (`router.py:102`) instead of Pydantic request models — a non-string `content` produces a 500 (Pydantic error inside the adapter) instead of a clean 422.

---

## 4. Scalability

The design is **single-replica by construction**; every piece of coordination state is in-process memory:

- `ModelEngine.active_lora` and `system_prompt` (`engine.py:25-26`) — a hot-swap on one replica leaves other replicas serving the old adapter; even on one replica the swap is non-atomic w.r.t. in-flight requests (a session can flip models between turns with no version pinning).
- `ContinuousLearner._lock` and `history`, `LoRATrainer._version`, `InteractionBuffer._threshold_fired_at`, telemetry `recorder` — none survive restart, none are shared across workers. Running `uvicorn --workers 2` would double-train and race the threshold.
- SQLite with a single locked connection is the write path for *every* message on the hot path (1.7); no WAL mode, no connection pool, and the promised "Postgres DSN later" (`buffer.py:6`) has no abstraction behind it — the code is `sqlite3`-specific.
- `buffer.fetch()` loads the entire interactions table into a Python list (`buffer.py:79-92`) — unbounded memory as history grows.
- One engine, one port, no request queueing/admission control, no per-session concurrency limits; the gateway will happily accept unbounded concurrent requests and funnel them into a 120 s-timeout client.
- No streaming from gateway to clients: `stream_chat` exists on the engine but no route exposes it, so long generations hold connections silently and time-to-first-token is worst-case.

---

## 5. End-to-end pipeline & tool abstraction (the headline promise)

### 5.1 The E2E loop exists as seams, not as a wired system

The advertised pipeline — gateway → guardrails → graph → engine → buffer → trainer → gate → hot-swap — is present as *composable parts*, and the seams (injectable `train_fn`, `runner`, `evaluator`) are genuinely good design. But the connective tissue is missing:

1. **No session memory at inference.** `CompiledGraph.as_handler` (`graph.py:143-160`) wraps each inbound message in a **fresh State with exactly one message**. `session_id` exists in the protocol, and the buffer stores full per-session history with an index built for it (`buffer.py:31`), yet nothing reads it back. Every conversation is amnesiac; the "memory" subsystem feeds training only. This is the single biggest functional gap: the framework cannot hold a multi-turn conversation out of the box.
2. **The agent loop is not a framework feature.** The example (`examples/basic_agent.py:29-40`) hand-writes `think`/`act` nodes, hand-parses tool JSON with `json.loads(s.messages[-1].content)` (no error handling — malformed model output crashes the request), and routes on `content.startswith('{"tool"')`. A framework claiming to abstract orchestration should ship a prebuilt `ToolNode`/agent-executor: registry lookup, `Tool.execute`, `ToolValidationError` → feedback-to-model loop, iteration bounds. All the primitives exist (`Tool.execute` validates and awaits, `graph.py` supports cycles); they are simply never composed.
3. **Tool calling is prompt-hacked instead of native.** `render_tool_prompt` (`omniai/graph/tools.py:102-110`) begs the model for JSON-only output, while both target backends support native OpenAI `tools=[...]` and guided/constrained decoding (vLLM `guided_json`, SGLang structured outputs) that would make schema conformance a *guarantee* instead of a hope. The engine's `chat` even passes `**kwargs` through — the capability is one parameter away.
4. **No tool registry.** Tools are loose `Tool` objects; there is no `ToolRegistry` to resolve `ToolCall.name → Tool`, no collision detection, no per-channel/per-session tool scoping, and the sandbox (`SandboxExecution`) is a "tool-style class" that is never actually exposed as a `Tool`.

### 5.2 Training data formatting will teach the model the wrong format

`_default_peft_train` (`learning.py:58-89`) builds each example as:

```python
{"text": f"{system}\n{prompt}\n{completion}"}
```

No chat template (`<|im_start|>` etc. for the Qwen model in the example), no EOS token, no prompt-loss masking (the model trains to predict the *prompt* too). The resulting adapter learns a format that disagrees with the one the serving stack uses at inference — so even with the gate fixed, adapters will genuinely regress and be rejected, or worse, marginally pass while degrading chat behavior. Use the tokenizer's `apply_chat_template` and completion-only loss (`DataCollatorForCompletionOnlyLM` / `SFTConfig` masking).

### 5.3 Abstraction-layer report card

| Layer | Grade | Notes |
| --- | --- | --- |
| Serving (`EngineConfig` + adapters) | **B+** | Genuinely backend-neutral; per-backend flag mapping incl. rejecting unsupported combos (`backends.py:100-101`) is the right pattern. Loses points for DEVNULL logs, no supervision, and untyped `quantization`/`kv_cache` strings (no enum validation → typos surface as backend boot failures with discarded logs). |
| Protocol (`OmniMessage`) | **B** | Right idea, one canonical type everywhere; but lossy `to_openai` (1.8), no content-parts (images), no streaming frames. |
| Graph | **B−** | Clean minimal core with bounded cycles; missing: replace-semantics for state, checkpointing/persistence, parallel branches, per-node retry policy, reachability validation. |
| Gateway | **C+** | Adapter-as-pure-codec is testable and nice; sinks all its goodwill on auth (2.1), Discord correctness (2.2), and non-optional serial observers (1.7). |
| Tools | **C+** | Schema generation from hints is solid; execution/registry/agent-loop layers absent (5.1). |
| Memory/learning | **C−** | The end-to-end concept is impressive on paper; bugs 1.1/1.2/1.4 mean it has plausibly never completed a real cycle. |
| Guardrails/evals/sandbox | **C** | Right *shapes* (policy objects, golden datasets, injectable runners), shallow implementations with the critical bugs noted above. |

### 5.4 What is genuinely good (credit where due)

- Clear layering with a single canonical message type; no circular dependencies; every module is import-light with optional heavy deps.
- Injectable seams everywhere (`train_fn`, `runner`, `evaluator`, `executor`) keep 910 lines of tests GPU-free and fast (68 tests in ~1 s).
- Eval-gated adapter deployment is the *correct* MLOps instinct — most frameworks at this maturity don't even attempt it.
- `traced_span` with a no-dependency fallback is a nice pattern (modulo the unbounded recorder).
- Honest config validation (rejecting `kv_cache` values a backend can't provide) rather than silent ignoring.

---

## 6. Self-hosted LLM deep-dive: the serving stack end-to-end

This is the framework's boldest promise — own the vLLM/SGLang lifecycle so users never touch backend details — and it is where the gap between the abstraction and real GPU operations is widest. Walking the actual self-hosted pipeline stage by stage:

### 6.1 Boot: model acquisition and readiness

- `vllm serve Qwen/Qwen2.5-7B-Instruct` downloads ~15 GB from Hugging Face on first boot. The adapter neither pre-fetches, configures `HF_HOME`/`HF_TOKEN` (gated models simply fail), nor scales the readiness timeout to download size — the default 300 s (`engine.py:45`) is routinely exceeded on first boot, and with stdout/stderr → DEVNULL (`backends.py:53-57`) the operator sees only "server did not become ready", not "downloading shard 3/8".
- **`subprocess.Popen` is called with no `env=`** (`backends.py:53`). Consequences:
  - `CUDA_VISIBLE_DEVICES` cannot be set per engine — two engines on one host fight over GPU 0;
  - **vLLM's dynamic LoRA endpoint is disabled by default**: `/v1/load_lora_adapter` requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` in the *server's* environment. The adapter never sets it, so on a stock vLLM the advertised "zero-downtime hot-swap" (`engine.py:131-140`) is rejected by the server — the continuous-learning loop is broken at the serving layer independently of bug 1.1.
- `/health` returning 200 does not mean *warm*: CUDA graph capture and cache priming make the first real request pay seconds of latency. There is no warm-up generation after readiness.
- No port-collision or GPU-availability preflight: `tensor_parallel_size=2` on a 1-GPU box, or a port already bound, both die invisibly in DEVNULL.

### 6.2 Hardware-optimization mapping is thinner than it looks

- `quantization` is a free string. AWQ/GPTQ require *pre-quantized checkpoints* — passing `quantization="awq"` with the fp16 Qwen checkpoint from the README crashes the server at load. FP8 is coupled to `--kv-cache-dtype fp8` unconditionally (`backends.py:96-97`), which is a policy decision (accuracy trade-off) the user never asked for and cannot opt out of.
- **The promised prefix-cache win is never switched on for vLLM.** `SkillLoader.install` + `set_system_prompt` are documented as "prefilled once and served from cache" (`skills.py:14-17`), but the vLLM adapter never passes `--enable-prefix-caching` — on vLLM's V0 engine (which `vllm>=0.5.0` permits) automatic prefix caching is **off by default**, so every request re-prefills the full skill prompt. The claimed RadixAttention benefit is real only on SGLang.
- Version drift is unmanaged: `vllm>=0.5.0` and `sglang>=0.3.0` (`pyproject.toml:22-23`) span years of releases in which these exact CLI flags and LoRA endpoints were added, renamed, or gated. The adapters hardcode flag spellings with no version detection and no startup verification that the flags exist — the failure mode is, again, a silent DEVNULL crash.
- No mapping for the knobs that matter most in self-hosted throughput tuning: `max_num_seqs`/`max-running-requests` (batch concurrency), dtype, seed, `swap-space`, pipeline parallelism, `served-model-name`.

### 6.3 LoRA lifecycle: load-only, capped at 4, no rollback

- `--max-loras 4` is **hardcoded** (`backends.py:109`). The continuous learner mints a new adapter every cycle (`...-lora-v1`, `-v2`, …) and there is **no unload path anywhere in the abstraction** — `BackendAdapter` defines `lora_load_endpoint` but no unload, and `ModelEngine` has no `unload_lora_adapter`. Adapters accumulate in the server until the cap/GPU memory is hit, at which point every subsequent cycle fails. The system's core loop has a built-in expiry of ~4 cycles.
- **No rollback:** after a bad swap there is no "reactivate previous adapter" operation and no record of what was previously active (`active_lora` is overwritten in place, `engine.py:139`). Combined with the broken gate (1.1), a regressing adapter goes live with no way back short of a restart.
- **Activation is a client-side fiction.** `active_lora` only changes which `model` string *this one `ModelEngine` instance* sends (`engine.py:83`). Any other client of the same server — a second gateway replica, the eval gate mid-flight, a monitoring probe — still gets the base model or its own idea of the adapter. There is no server-side notion of "the production adapter", no alias/routing table, and no atomicity: requests in flight during a swap straddle two models within one session.
- Adapter artifacts are **local filesystem paths** handed to the server (`load_lora_adapter(name, path)`): this silently assumes trainer and inference server share a filesystem. The moment serving runs in a container or on another node (i.e., any real deployment), the path is meaningless. Self-hosted MLOps needs an artifact store (S3/NFS) plus a registry mapping adapter name → version → artifact → eval verdict; none exists (the `history` list on the learner is in-memory and unshared).

### 6.4 Training and serving fight for the same GPU

`LoRATrainer` defaults to a `ProcessPoolExecutor` **on the same host** (`learning.py:119-121`), and `_default_peft_train` loads the full base model for SFT. Meanwhile vLLM has pre-allocated ~90% of GPU memory (its default `gpu_memory_utilization`). The default topology of the flagship feature is therefore: *serving holds the GPU, training OOMs* — or, if the box has spare GPUs, training grabs GPU 0 alongside vLLM anyway because nobody sets `CUDA_VISIBLE_DEVICES` (6.1). There is no device placement, no training queue, no option to schedule training off-peak or on a separate node. For a credible self-hosted story, training must be a dispatchable job (separate worker/node, or at minimum an explicit device map and admission check), not an in-process pool.

### 6.5 Request path: no token budgeting, no admission control, no streaming out

- **Nothing counts tokens.** The skill prompt is unbounded (`compose_system_prompt` concatenates every skill), conversation state is append-only (3), and `max_model_len` is never checked client-side — once a session's history exceeds the context window, every subsequent request 400s forever (a permanently bricked session, since history is never truncated). A tokenizer-aware budget (truncate/summarize to fit `max_model_len` minus generation headroom) is table stakes for self-hosted serving where *you* chose the context length.
- The gateway accepts unbounded concurrency and funnels it into one httpx client with a flat 120 s timeout; the server's real capacity (`max_num_seqs`, KV-cache pressure) is invisible because **vLLM's `/metrics` (queue depth, KV-cache utilization, TTFT) is never scraped** — the framework flies blind past the exact signals self-hosting exists to expose. No queueing, no load shedding, no per-session limits.
- `stream_chat` exists but no gateway route uses it, so self-hosted TTFT advantages are thrown away; eval-gate traffic also shares the production server (no shadow/offline eval path), so every learning cycle degrades live latency.

### 6.6 Shutdown and supervision of GPU processes

- `stop()` sends SIGTERM to the *parent* process only (`backends.py:60-67`). With `tensor_parallel_size>1`, vLLM spawns worker processes (multiprocessing/Ray) holding NCCL communicators; killing the parent can orphan workers that **keep GPU memory allocated**, wedging the GPU until manual cleanup. The subprocess should be started in its own process group (`start_new_session=True`) and killed with `killpg`, with a post-mortem check that GPU memory was actually released.
- There is no supervision loop: if the server OOMs or segfaults mid-run (routine events in GPU serving), `self.process` still holds a dead PID, `/health` on the gateway still says "ok" (3), and every request 500s until a human intervenes. A restart policy with crash-loop backoff — and re-loading the previously active adapter after restart (which requires the registry from 6.3) — is the minimum viable supervisor.

### 6.7 What a credible self-hosted E2E pipeline needs (target architecture)

```
            ┌────────────┐   scrape /metrics   ┌───────────────┐
 clients ──▶│  Gateway    │◀───────────────────│  Supervisor    │
            │ (auth, SSE, │                     │ (restart, warm │
            │  admission) │                     │  -up, GPU mem) │
            └─────┬──────┘                     └───────┬───────┘
                  ▼ token-budgeted requests            ▼ owns lifecycle
            ┌────────────────────────────────────────────────┐
            │  vLLM / SGLang  (env-configured, pinned flags)  │
            │  adapter alias table: "prod" → skills-v7        │
            └───────────────▲────────────────────────────────┘
                            │ load/unload/activate via registry
            ┌───────────────┴───────────────┐
            │  Adapter Registry (DB + S3)    │◀── eval verdicts (shadow eval,
            │  name → version → artifact     │    temperature=0, base baseline)
            └───────────────▲───────────────┘
                            │ artifact push
            ┌───────────────┴───────────────┐
            │  Training worker (own GPU/node,│◀── watermarked batches from
            │  chat-template SFT, queued)    │    interaction store
            └───────────────────────────────┘
```

The current codebase has the right *interfaces* for roughly half of these boxes; the missing pieces are the registry, the supervisor, device placement, token budgeting, and metrics-driven admission — detailed in the roadmap below.

---

## 7. Prioritized roadmap

### P0 — correctness & safety (do before any real traffic)
1. Fix eval ordering: load adapter (inactive) → score → activate or unload (1.1).
2. Hold task references + `add_done_callback` with logging on `trigger`; introduce `logging` across the codebase (1.2).
3. Kill the *container* on sandbox timeout; pass code via stdin; add `--pids-limit`, output caps (1.3).
4. Add authentication (API keys/JWT dependency in FastAPI), rate limiting, and Discord Ed25519 verification + PING handshake (2.1, 2.2).
5. Watermark training data; persist trainer version; make threshold accounting transactional and post-success (1.4, 1.10).
6. Pin the eval gate: explicit base-model baseline, `temperature=0`, load-then-score (1.5).
7. **Serving-layer unblockers for the learning loop:** pass an `env` dict through `BackendAdapter.start()` (set `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `CUDA_VISIBLE_DEVICES`, `HF_HOME`); add `unload_lora_adapter` + previous-adapter tracking for rollback; make `--max-loras` configurable and evict old adapters each cycle (6.1, 6.3).
8. Capture backend stdout/stderr to files/logger and kill the whole process group on `stop()` (`start_new_session=True` + `killpg`) so TP workers can't wedge the GPU (6.1, 6.6).

### P1 — reliability & the missing product core
9. Session memory: `as_handler` hydrates State from the buffer by `session_id` — with a **tokenizer-aware budget** that truncates/summarizes to fit `max_model_len` (6.5).
10. Ship a `ToolRegistry` + prebuilt `ToolNode`/agent loop; adopt native `tools=` + guided decoding (vLLM `guided_json` / SGLang structured output — a self-hosting advantage the framework currently ignores); fix `to_openai` to carry `tool_calls`/`tool_call_id`.
11. Engine resilience: retries with backoff + jitter, circuit breaker, health-aware `/health` that probes the backend, a supervisor with crash-loop backoff that re-loads the active adapter after restart, warm-up generation after readiness, graceful shutdown via FastAPI lifespan (6.1, 6.6).
12. Serving correctness: enable `--enable-prefix-caching` on vLLM (or verify V1 default) so the skill-prompt caching claim is real; validate `quantization` against checkpoint format; decouple the forced fp8 KV-cache; pin and version-detect backend flags (6.2).
13. Move training off the serving GPU: explicit device placement at minimum, a queued training worker (separate process with its own `CUDA_VISIBLE_DEVICES`, or separate node) as the default topology (6.4).
14. Move observers off the request path (`asyncio.create_task` + bounded queue); log blocked messages to an audit table pre-redaction (encrypted).
15. Correct SFT formatting: chat template + EOS + completion-only loss; add data quality filters (dedup, self-output exclusion, feedback signal).

### P2 — scale-out
16. Adapter registry + artifact store (DB + S3/NFS): name → version → artifact → eval verdict, server-side "prod" alias for atomic activation across replicas, rollback as a first-class operation (6.3, 6.7).
17. Externalize coordination state (Postgres/Redis): buffer behind a real repository interface, distributed threshold/locking so N replicas work.
18. Metrics-driven operation: scrape vLLM/SGLang `/metrics` (KV-cache utilization, queue depth, TTFT) into telemetry; admission control and load shedding keyed to it; per-session concurrency limits (6.5).
19. Expose streaming end-to-end (SSE on REST, frames on WS) to bank the self-hosted TTFT advantage; separate shadow-eval path so gate traffic doesn't share the production server (6.5).
20. Data governance: retention TTL, consent flags, erasure path for training data.
21. CI (pytest + ruff + mypy), Dockerfile with pinned backend versions, config from environment.

---

*Test suite status at review time: `68 passed in 1.07s` on Python 3.11.*
