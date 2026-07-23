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

## 6. Prioritized roadmap

### P0 — correctness & safety (do before any real traffic)
1. Fix eval ordering: load adapter (inactive) → score → activate or unload (1.1).
2. Hold task references + `add_done_callback` with logging on `trigger`; introduce `logging` across the codebase (1.2).
3. Kill the *container* on sandbox timeout; pass code via stdin; add `--pids-limit`, output caps (1.3).
4. Add authentication (API keys/JWT dependency in FastAPI), rate limiting, and Discord Ed25519 verification + PING handshake (2.1, 2.2).
5. Watermark training data; persist trainer version; make threshold accounting transactional and post-success (1.4, 1.10).
6. Pin the eval gate: explicit base-model baseline, `temperature=0`, load-then-score (1.5).

### P1 — reliability & the missing product core
7. Session memory: `as_handler` hydrates State from the buffer by `session_id` (with a window/summarization policy).
8. Ship a `ToolRegistry` + prebuilt `ToolNode`/agent loop; adopt native `tools=` + guided decoding; fix `to_openai` to carry `tool_calls`/`tool_call_id`.
9. Engine resilience: retries with backoff + jitter, circuit breaker, health-aware `/health`, subprocess log capture + supervision/restart, graceful shutdown via FastAPI lifespan.
10. Move observers off the request path (`asyncio.create_task` + bounded queue); log blocked messages to an audit table pre-redaction (encrypted).
11. Correct SFT formatting: chat template + EOS + completion-only loss; add data quality filters (dedup, self-output exclusion, feedback signal).

### P2 — scale-out
12. Externalize coordination state (Postgres/Redis): buffer behind a real repository interface, adapter registry with versioned activation, distributed threshold/locking so N replicas work.
13. Expose streaming end-to-end (SSE on REST, frames on WS); admission control and per-session concurrency limits.
14. Data governance: retention TTL, consent flags, erasure path for training data.
15. CI (pytest + ruff + mypy), Dockerfile, pinned lockfile, config from environment.

---

*Test suite status at review time: `68 passed in 1.07s` on Python 3.11.*
