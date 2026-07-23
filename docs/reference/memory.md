# omniai.memory

## `InteractionBuffer`

```python
InteractionBuffer(
    database_url="sqlite+aiosqlite:///interactions.db",  # any SQLAlchemy async URL, or a plain path (treated as SQLite)
    threshold=None,                                      # fire on_threshold every N logged messages
    on_threshold=None,                                   # sync/async callable
)
```

- `await log(message: OmniMessage)` — persist (idempotent by message id); O(1) threshold accounting.
- `await count() -> int` — real row count.
- `await fetch(session_id=None, limit=None, since: datetime | None = None) -> list[dict]` — oldest-first dicts (`id, session_id, channel, role, content, tool_calls, metadata, created_at`); `since` filters to rows after the (naive-UTC) timestamp.
- `await aclose()` / `close()` — dispose the engine.
- Instances are async-callable, so a buffer can be passed directly as a gateway observer.

Schema: `Interaction` SQLModel table (`omniai.memory.models`); migrations via Alembic ([guide](../how_to/database_migrations.md)).

## `SkillLoader`

```python
SkillLoader(preamble="You are a capable assistant with the following skills.")
```

`load_file(path)`, `load_directory(dir)` (globs `*.skill.md` / `skill.md`; frontmatter `name:`/`description:` + body), `compose_system_prompt()`, `install(engine)` (calls `engine.set_system_prompt`). `Skill` model: `name`, `description`, `body`, `source`, `render()`.

## `format_training_pairs`

```python
format_training_pairs(logs: list[dict], system_prompt=None) -> list[{"prompt", "completion", "system"?}]
```

Pairs each user turn with the next assistant turn per session; tool outputs fold into the prompt as `[tool output] ...` lines.

## `LoRATrainer`

```python
LoRATrainer(base_model, output_root="adapters", train_fn=None, executor=None, **hyperparams)
```

`await train(pairs) -> (adapter_name, adapter_path)` — runs `train_fn` (default: PEFT/TRL SFT; requires the `training` extra) in a process pool (or injected executor). Adapter names are timestamped + random-suffixed (restart-safe). Empty `pairs` raises `ValueError`.

## `ContinuousLearner`

```python
ContinuousLearner(buffer, trainer, engine=None, evaluator=None, min_pairs=1, since=None)
```

- `await run_cycle() -> report` — fetch since high-water mark → pair → train → (evaluate) → hot-swap; reports `{"status": "deployed" | "rejected" | "skipped", ...}`. Serialized by an internal lock; the high-water mark advances once data is trained.
- `trigger()` — fire-and-forget task; suitable as `buffer.on_threshold`.
- `history: list[dict]`, `on_report` hook, `high_water: datetime | None`.
