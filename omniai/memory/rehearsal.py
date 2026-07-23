"""Rehearsal buffer: mix golden general-purpose data into every SFT run.

Fine-tuning repeatedly on narrow business data erodes the base model's
general language and reasoning ability (catastrophic forgetting). The
rehearsal buffer holds a fixed "golden" set of diverse, high-quality general
prompts and mixes them into every training batch so each adapter keeps
rehearsing the basics. The default mix keeps new business data at 20% of the
batch (``new_data_ratio=0.2``), i.e. four rehearsal pairs for every new pair,
capped by how much golden data is available.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class RehearsalBuffer:
    """Golden general-purpose pairs mixed into every training run."""

    def __init__(
        self,
        pairs: list[dict[str, str]],
        new_data_ratio: float = 0.2,
        seed: int | None = None,
    ):
        if not 0 < new_data_ratio <= 1:
            raise ValueError("new_data_ratio must be in (0, 1]")
        self.pairs = list(pairs)
        self.new_data_ratio = new_data_ratio
        self._rng = random.Random(seed)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, new_data_ratio: float = 0.2, seed: int | None = None
    ) -> RehearsalBuffer:
        """Load golden pairs from JSONL rows of {"prompt", "completion"}
        (or the equivalent {"instruction", "output"})."""
        pairs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append(
                {
                    "prompt": row.get("prompt", row.get("instruction", "")),
                    "completion": row.get("completion", row.get("output", "")),
                }
            )
        return cls(pairs, new_data_ratio=new_data_ratio, seed=seed)

    def mix(self, new_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        """Blend new pairs with sampled golden pairs and shuffle.

        With the default ratio of 0.2, ``len(new_pairs)`` new examples pull
        in up to 4x as many golden examples.
        """
        if not new_pairs:
            return []
        rehearsal_target = round(len(new_pairs) * (1 - self.new_data_ratio) / self.new_data_ratio)
        rehearsal = self._rng.sample(self.pairs, min(rehearsal_target, len(self.pairs)))
        mixed = list(new_pairs) + rehearsal
        self._rng.shuffle(mixed)
        return mixed
