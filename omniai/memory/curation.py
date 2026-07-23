"""Data curation: LLM-as-a-judge screening of interaction logs (anti-poisoning).

Training on raw chat logs teaches the model its own mistakes ("model
collapse"). :class:`InteractionJudge` is the background curator: it turns raw
logs into candidate instruction pairs, has a judge model (ideally a heavier,
smarter one than the model being trained) score each pair, and keeps only the
pairs that clear ``min_score``. Explicit user feedback recorded in message
metadata short-circuits the judge: thumbs-down pairs are dropped outright,
thumbs-up pairs are kept.

The judge callable is injectable, so tests — and deployments that prefer a
reward model or heuristics — never need a live LLM.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from omniai.memory.learning import format_training_pairs

logger = logging.getLogger("omniai.curation")

# judge_fn(instruction, output) -> score in [0, 1] (sync or async)
JudgeFn = Callable[[str, str], float | Awaitable[float]]

JUDGE_PROMPT = """You are a strict data-quality judge for LLM fine-tuning.
Rate the assistant response below for factual accuracy, instruction-following,
and formatting. Penalize hallucinations, refusals-by-confusion, and errors
heavily.

[Instruction]
{instruction}

[Response]
{output}

Reply with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<short>"}}"""


def _extract_score(text: str) -> float:
    """Parse the judge's score, tolerating markdown fences and chatter."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return float(json.loads(text)["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r'"score"\s*:\s*([01](?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
    return 0.0  # unparseable verdict: never train on it


class InteractionJudge:
    """Curates interaction logs into high-quality training pairs.

    Parameters
    ----------
    engine:
        Engine whose ``chat_text`` runs the judge prompt. Point this at a
        stronger model than the one being trained. Ignored when ``judge_fn``
        is given.
    judge_fn:
        Direct scoring callable ``(instruction, output) -> float`` — the
        injectable seam for tests, reward models, or heuristics.
    min_score:
        Pairs scoring below this are discarded.
    model:
        Optional model override passed to ``engine.chat_text`` (e.g. a larger
        judge model served by the same engine).
    """

    def __init__(
        self,
        engine=None,
        judge_fn: JudgeFn | None = None,
        min_score: float = 0.7,
        model: str | None = None,
    ):
        if engine is None and judge_fn is None:
            raise ValueError("InteractionJudge needs an engine or a judge_fn")
        self.engine = engine
        self.judge_fn = judge_fn
        self.min_score = min_score
        self.model = model

    async def score(self, instruction: str, output: str) -> float:
        if self.judge_fn is not None:
            result = self.judge_fn(instruction, output)
            if hasattr(result, "__await__"):
                result = await result
            return float(result)
        kwargs: dict[str, Any] = {"temperature": 0}
        if self.model:
            kwargs["model"] = self.model
        prompt = JUDGE_PROMPT.format(instruction=instruction, output=output)
        text = await self.engine.chat_text([{"role": "user", "content": prompt}], **kwargs)
        return _extract_score(text)

    @staticmethod
    def _feedback(pair: dict[str, str]) -> str | None:
        """Explicit user feedback attached during pair formatting, if any."""
        value = pair.get("feedback")
        return str(value).lower() if value is not None else None

    async def curate(self, logs: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Raw interaction logs -> judged, trainer-ready pairs."""
        candidates = format_training_pairs(logs)
        kept: list[dict[str, str]] = []
        dropped = 0
        for pair in candidates:
            feedback = self._feedback(pair)
            if feedback in ("negative", "thumbs_down", "bad"):
                dropped += 1
                continue
            if feedback in ("positive", "thumbs_up", "good"):
                kept.append({"prompt": pair["prompt"], "completion": pair["completion"]})
                continue
            score = await self.score(pair["prompt"], pair["completion"])
            if score >= self.min_score:
                kept.append({"prompt": pair["prompt"], "completion": pair["completion"]})
            else:
                dropped += 1
        logger.info("curation kept %d/%d pairs", len(kept), len(candidates))
        return kept
