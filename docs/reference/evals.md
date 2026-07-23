# omniai.evals

## `GoldenCase` (dataclass)

`prompt: str`, `expected_tool: str`, `expected_args: dict | None = None` (`None` accepts any arguments). `matches(tool_name, args) -> bool`.

## `GoldenDataset` (dataclass)

`cases: list[GoldenCase]`; `GoldenDataset.from_jsonl(path)` loads one JSON object per line (`prompt`, `expected_tool`, optional `expected_args`).

## `AdapterGate`

```python
AdapterGate(engine, dataset, baseline_accuracy=None, tolerance=0.0)
```

- `await score(model=None) -> (accuracy, failed_prompts)` — runs every golden prompt through `engine.chat_text`; replies must be `{"tool": ..., "arguments": ...}` JSON (code fences tolerated).
- `await establish_baseline() -> float` — scores the base model; stored as the bar.
- `await evaluate(adapter_name, adapter_path="") -> EvalVerdict` — establishes the baseline if needed, scores the candidate (adapter name as model), accepts iff `accuracy >= baseline - tolerance`.
- `evaluator` property — bound callable matching `ContinuousLearner(evaluator=...)`.

## `EvalVerdict` (dataclass)

`accepted: bool`, `accuracy: float`, `baseline: float`, `adapter: str`, `failures: list[str]` (the prompts that failed).
