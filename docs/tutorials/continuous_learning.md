# Continuous learning

Turn your interaction log into LoRA adapters that improve the live model — trained in the background, gated by evals, and hot-swapped with zero downtime.

**Prerequisites:** [Multi-channel chatbot](multi_channel_chatbot.md) (you need an `InteractionBuffer` receiving traffic) · concepts: [memory & learning](../concepts/memory_and_learning.md) · install: `pip install -e ".[training]"`

## The cycle

1. The gateway logs every user/tool/assistant message into the buffer.
2. When a threshold is crossed (or an admin triggers it), the learner fetches **only interactions since the last trained cycle** (a high-water mark keeps cycles bounded).
3. Logs are paired into instruction-tuning examples (`format_training_pairs` matches each user turn with the next assistant turn, folding tool outputs into the prompt).
4. A PEFT LoRA adapter trains off the event loop (process pool by default).
5. The **eval gate** scores the candidate on a golden dataset and rejects regressions.
6. Accepted adapters are hot-swapped into the serving engine via its REST API (`/v1/load_lora_adapter` on vLLM) — no restart.

## Wire it up

```python
from omniai.engine import ModelEngine
from omniai.evals import AdapterGate, GoldenDataset
from omniai.memory import ContinuousLearner, InteractionBuffer, LoRATrainer

engine = ModelEngine.create({"model": "Qwen/Qwen2.5-7B-Instruct", "backend": "vllm"})
await engine.start()

buffer = InteractionBuffer("interactions.db", threshold=1000)   # fire every 1000 messages
gate = AdapterGate(engine, GoldenDataset.from_jsonl("golden.jsonl"))

learner = ContinuousLearner(
    buffer,
    LoRATrainer(engine.config.model, output_root="adapters"),
    engine=engine,
    evaluator=gate.evaluator,     # candidates must not regress below baseline
    min_pairs=50,                 # skip cycles with too little new data
)
buffer.on_threshold = learner.trigger   # automatic; or: await learner.run_cycle() manually
```

## The eval gate

`golden.jsonl` holds prompts and the tool call each must produce:

```json
{"prompt": "weather in Paris?", "expected_tool": "get_weather", "expected_args": {"city": "Paris"}}
{"prompt": "search for llamas", "expected_tool": "web_search"}
```

The gate scores the base model once to establish a baseline, then scores every candidate adapter; a candidate below `baseline - tolerance` is rejected and **never reaches the engine**. Reports land in `learner.history`:

```python
{"status": "deployed", "adapter": "Qwen2.5-7B-Instruct-lora-20260723...-a1b2", "pairs": 812}
{"status": "rejected", "reason": "failed_eval_gate", ...}
```

## Operational notes

- Adapter names are timestamped and restart-safe; the newest deployed adapter is re-applied automatically if the engine subprocess crashes and is restarted by the supervisor.
- Cycles are serialized by a lock — overlapping triggers can't start concurrent trainings.
- Training pairs inherit the engine's current system prompt so adapters learn in-context behavior.
- For custom training loops (DPO, different SFT configs), inject `LoRATrainer(train_fn=...)`; tests use this seam to run without GPUs.

## Next steps

- [Evaluate adapters](../how_to/evaluate_adapters.md) — building golden datasets and tolerances.
- [LoRA hot-swap](../how_to/lora_hot_swap.md) — the mechanics of zero-downtime adapter loading.
- [Database migrations](../how_to/database_migrations.md) — the interaction log's schema lifecycle.
