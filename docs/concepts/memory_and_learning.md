# Memory & learning

OmniAI treats interaction history as a training asset: every conversation is logged, and logs periodically become LoRA adapters that improve the live model — safely.

## The interaction buffer

`InteractionBuffer` is a gateway observer: every inbound user message, tool output, and assistant reply is persisted asynchronously. Storage is SQLModel over SQLAlchemy's async engine — **database-agnostic by URL** (Postgres in production, SQLite in dev, anything async-capable). Design points:

- Logging is O(1) per message: the training threshold uses an incrementally maintained counter, not a `COUNT(*)` per insert.
- Writes are idempotent by message ID (`merge` semantics) — replays don't duplicate.
- The threshold callback (`on_threshold`) is how the learner is triggered automatically.

## Skills

`SkillLoader` parses `skill.md` files (frontmatter + body) at boot and composes them into a system prompt installed via `engine.set_system_prompt` — cheap at inference time thanks to [prefix caching](serving_engines.md).

## The learning cycle

`ContinuousLearner.run_cycle()` is a pipeline with three safety properties:

1. **Bounded**: it fetches only interactions past a *high-water mark* (the timestamp of the last trained cycle), so cycle cost tracks new traffic, not table size. `format_training_pairs` pairs each user turn with the next assistant turn in the same session, folding tool outputs into the prompt.
2. **Off the event loop**: LoRA training (PEFT SFT by default, injectable `train_fn`) runs in a process pool; the API keeps serving.
3. **Gated**: if an evaluator is wired (see [adapter evals](../how_to/evaluate_adapters.md)), a candidate that regresses below baseline is rejected and never reaches the engine.

Accepted adapters are [hot-swapped](../how_to/lora_hot_swap.md) live. Cycles are serialized by a lock; adapter names are timestamped and restart-safe; every cycle emits a report (`deployed` / `rejected` / `skipped`) into `learner.history` and the `on_report` hook (wired to Prometheus in production).

## Trade-offs made explicit

- Incremental fetching can drop a pair whose user turn and assistant turn straddle a cycle boundary — accepted for bounded cycles.
- Each adapter trains on data since the previous cycle. If you want cumulative retraining, seed `ContinuousLearner(since=None)` and manage the mark yourself.
- The high-water mark is in-memory; after a process restart the next cycle re-reads from the seeded `since` (or everything). Persist it externally if that matters to your volume.
