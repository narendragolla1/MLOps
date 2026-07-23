# How to evaluate LoRA adapters before deployment

The eval gate is the CI/CD step for the [continuous-learning loop](../tutorials/continuous_learning.md): every candidate adapter is scored on a golden dataset, and anything that regresses below baseline is rejected before it can be hot-swapped.

## Build a golden dataset

Each case is a prompt plus the tool call it must produce (`expected_args: null` accepts any arguments):

```json
{"prompt": "weather in Paris?", "expected_tool": "get_weather", "expected_args": {"city": "Paris"}}
{"prompt": "search for llamas", "expected_tool": "web_search"}
```

```python
from omniai.evals import GoldenDataset, GoldenCase

dataset = GoldenDataset.from_jsonl("golden.jsonl")
# or in code: GoldenDataset(cases=[GoldenCase("weather in Paris?", "get_weather", {"city": "Paris"})])
```

Pick cases that cover every tool, include near-miss prompts (where the wrong tool is tempting), and prompts that should produce *no* tool call if that matters to you.

## Gate an adapter

```python
from omniai.evals import AdapterGate

gate = AdapterGate(engine, dataset, tolerance=0.0)   # tolerance: allowed drop vs baseline
verdict = await gate.evaluate("candidate-adapter-name")
verdict.accepted, verdict.accuracy, verdict.baseline, verdict.failures
```

The first evaluation scores the **base model** to establish the baseline; each candidate is then scored with its adapter name as the model (how vLLM routes to loaded adapters). Responses must be tool-call JSON (`{"tool": ..., "arguments": ...}`); non-conforming answers count as failures.

## Wire into the learner

```python
learner = ContinuousLearner(buffer, trainer, engine=engine, evaluator=gate.evaluator)
```

Rejected adapters produce a `{"status": "rejected", "reason": "failed_eval_gate", ...}` report and never touch the engine.

## Choosing `tolerance`

`0.0` (default) means "never worse than baseline" — right for tool-calling accuracy. A small tolerance (e.g. `0.05`) trades a little accuracy headroom for adapter freshness; justify it with a large enough dataset that 5% is signal, not one flaky case.
