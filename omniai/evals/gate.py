"""Golden-dataset eval gate for freshly trained LoRA adapters.

Every candidate adapter is scored on tool-calling accuracy against a golden
prompt suite, and the gate rejects any adapter whose accuracy falls below
the baseline (minus a configurable tolerance). Plug ``AdapterGate.evaluator``
into ``ContinuousLearner(evaluator=...)`` so a bad adapter can never be
hot-swapped into production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenCase:
    """One golden test: a prompt and the tool call it must produce."""

    prompt: str
    expected_tool: str
    expected_args: dict[str, Any] | None = None  # None = any args accepted

    def matches(self, tool_name: str | None, args: dict[str, Any] | None) -> bool:
        if tool_name != self.expected_tool:
            return False
        if self.expected_args is None:
            return True
        return args == self.expected_args


@dataclass
class GoldenDataset:
    cases: list[GoldenCase] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GoldenDataset:
        cases = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append(
                GoldenCase(
                    prompt=row["prompt"],
                    expected_tool=row["expected_tool"],
                    expected_args=row.get("expected_args"),
                )
            )
        return cls(cases)


@dataclass
class EvalVerdict:
    accepted: bool
    accuracy: float
    baseline: float
    adapter: str
    failures: list[str] = field(default_factory=list)


def _parse_tool_call(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract {"tool": ..., "arguments": ...} from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    return data.get("tool"), data.get("arguments")


class AdapterGate:
    """Scores adapters on the golden suite and enforces the baseline bar.

    ``engine.chat_text`` is called once per golden case with the candidate
    adapter activated; responses must be schema-conforming tool-call JSON.
    """

    def __init__(
        self,
        engine,
        dataset: GoldenDataset,
        baseline_accuracy: float | None = None,
        tolerance: float = 0.0,
    ):
        self.engine = engine
        self.dataset = dataset
        self.baseline_accuracy = baseline_accuracy
        self.tolerance = tolerance

    async def score(self, model: str | None = None) -> tuple[float, list[str]]:
        """Run the golden suite; returns (accuracy, failed prompts)."""
        if not self.dataset.cases:
            return 1.0, []
        failures: list[str] = []
        for case in self.dataset.cases:
            kwargs = {"model": model} if model else {}
            text = await self.engine.chat_text([{"role": "user", "content": case.prompt}], **kwargs)
            tool_name, args = _parse_tool_call(text)
            if not case.matches(tool_name, args):
                failures.append(case.prompt)
        accuracy = 1.0 - len(failures) / len(self.dataset.cases)
        return accuracy, failures

    async def establish_baseline(self) -> float:
        """Score the base model; stored as the bar candidates must clear."""
        self.baseline_accuracy, _ = await self.score()
        return self.baseline_accuracy

    async def evaluate(self, adapter_name: str, adapter_path: str = "") -> EvalVerdict:
        """Score a candidate adapter and accept/reject it vs the baseline."""
        if self.baseline_accuracy is None:
            await self.establish_baseline()
        baseline = self.baseline_accuracy
        assert baseline is not None  # establish_baseline always sets it
        accuracy, failures = await self.score(model=adapter_name)
        accepted = accuracy >= baseline - self.tolerance
        return EvalVerdict(
            accepted=accepted,
            accuracy=accuracy,
            baseline=baseline,
            adapter=adapter_name,
            failures=failures,
        )

    @property
    def evaluator(self):
        """Callable matching ContinuousLearner's ``evaluator`` seam."""
        return self.evaluate
