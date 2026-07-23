"""Continuous learning: interaction logs -> LoRA adapter -> zero-downtime swap.

Pipeline (see :class:`ContinuousLearner.run_cycle`):
  1. pull only *new* interaction logs (after the persisted watermark),
  2. curate them (LLM-as-a-judge / feedback filter) into instruction pairs,
  3. mix in rehearsal (golden) data to prevent catastrophic forgetting,
  4. train a PEFT LoRA adapter off the event loop (thread or subprocess),
  5. shadow-gate the adapter: load it inactive, score it, and only then
     activate it — rejected adapters are unloaded and their weights destroyed,
  6. advance the watermark only after a successful deployment.

The heavy Hugging Face training runs only when ``peft``/``trl`` are
installed; a custom ``train_fn`` can be injected for tests or bespoke loops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omniai.telemetry import traced_span

logger = logging.getLogger("omniai.learning")


def format_training_pairs(
    logs: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Convert interaction logs into instruction-tuning pairs.

    Pairs each user message with the next assistant message in the same
    session; tool outputs in between are folded into the prompt as context.
    Explicit user feedback in the assistant message's metadata is carried
    through on the pair (``feedback`` key) for curators to act on.
    """
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in logs:
        by_session.setdefault(row["session_id"], []).append(row)

    pairs: list[dict[str, str]] = []
    for rows in by_session.values():
        prompt: str | None = None
        context: list[str] = []
        for row in rows:
            role = row["role"]
            if role == "user":
                prompt = row["content"]
                context = []
            elif role == "tool" and prompt is not None:
                context.append(f"[tool output] {row['content']}")
            elif role == "assistant" and prompt is not None:
                instruction = prompt if not context else prompt + "\n" + "\n".join(context)
                pair = {"prompt": instruction, "completion": row["content"]}
                if system_prompt:
                    pair["system"] = system_prompt
                feedback = _row_feedback(row)
                if feedback is not None:
                    pair["feedback"] = feedback
                pairs.append(pair)
                prompt, context = None, []
    return pairs


def _row_feedback(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if isinstance(metadata, dict) and metadata.get("feedback") is not None:
        return str(metadata["feedback"])
    return None


def _default_peft_train(
    base_model: str, pairs: list[dict[str, str]], output_dir: str, **hp: Any
) -> str:
    """SFT LoRA training with Hugging Face peft/trl. Runs in a subprocess."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list(
        [{"text": f"{p.get('system', '')}\n{p['prompt']}\n{p['completion']}"} for p in pairs]
    )
    peft_config = LoraConfig(
        r=hp.get("lora_r", 16),
        lora_alpha=hp.get("lora_alpha", 32),
        lora_dropout=hp.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=hp.get("epochs", 1),
            per_device_train_batch_size=hp.get("batch_size", 2),
            learning_rate=hp.get("learning_rate", 2e-4),
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


class LoRATrainer:
    """Runs LoRA training off the event loop (process pool by default)."""

    def __init__(
        self,
        base_model: str,
        output_root: str | Path = "adapters",
        train_fn: Callable[..., str] | None = None,
        executor: Executor | None = None,
        **hyperparams: Any,
    ):
        self.base_model = base_model
        self.output_root = Path(output_root)
        self.train_fn = train_fn or _default_peft_train
        self.executor = executor
        self.hyperparams = hyperparams

    def _adapter_name(self) -> str:
        # Timestamp + random suffix: unique across process restarts, unlike
        # an in-memory version counter (which would collide after a restart)
        # or a disk scan (which collides whenever train_fn hasn't written
        # anything yet — e.g. a stub in tests, or a crash before save).
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{Path(self.base_model).name}-lora-{stamp}-{uuid.uuid4().hex[:4]}"

    async def train(self, pairs: list[dict[str, str]]) -> tuple[str, str]:
        """Train an adapter; returns (adapter_name, adapter_path)."""
        if not pairs:
            raise ValueError("No training pairs; refusing to train an empty adapter")
        name = self._adapter_name()
        output_dir = str(self.output_root / name)
        loop = asyncio.get_running_loop()
        executor = self.executor
        own_executor = False
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=1)
            own_executor = True
        try:
            with traced_span("memory.lora_train", {"adapter": name, "pairs": len(pairs)}):
                path = await loop.run_in_executor(
                    executor,
                    _train_call,
                    self.train_fn,
                    self.base_model,
                    pairs,
                    output_dir,
                    self.hyperparams,
                )
        finally:
            if own_executor:
                executor.shutdown(wait=False)
        return name, path


def _train_call(
    fn: Callable[..., str],
    base_model: str,
    pairs: list[dict[str, str]],
    output_dir: str,
    hp: dict[str, Any],
) -> str:
    # Module-level shim so the callable pickles cleanly into a subprocess.
    return fn(base_model, pairs, output_dir, **hp)


class ContinuousLearner:
    """Orchestrates the log -> curate -> train -> shadow-gate -> swap cycle.

    Wire ``learner.trigger`` as the buffer's ``on_threshold`` callback for
    automatic cycles, or call ``await learner.run_cycle()`` from an admin
    endpoint for manual triggering. Cycles are serialized by a lock so
    overlapping triggers cannot start concurrent trainings.

    Parameters
    ----------
    curator:
        Optional async ``logs -> pairs`` callable (e.g.
        :class:`~omniai.memory.curation.InteractionJudge`'s ``curate``) that
        filters raw logs into high-quality pairs. Defaults to uncurated
        :func:`format_training_pairs`.
    rehearsal:
        Optional :class:`~omniai.memory.rehearsal.RehearsalBuffer` mixed into
        every training batch against catastrophic forgetting.
    evaluator:
        Optional async ``(name, path) -> verdict`` scoring callable. When
        set, the adapter is **shadow-loaded** (inactive) before scoring and
        activated only on acceptance; rejected adapters are unloaded and
        their weights destroyed (``destroy_rejected``). This ordering matters:
        scoring an adapter that was never loaded onto the serving backend
        would silently fall back to the base model instead of testing the
        candidate.
    """

    def __init__(
        self,
        buffer,
        trainer: LoRATrainer,
        engine=None,
        evaluator: Callable[[str, str], Any] | None = None,
        curator: Callable[[list[dict[str, Any]]], Any] | None = None,
        rehearsal=None,
        min_pairs: int = 1,
        destroy_rejected: bool = True,
    ):
        self.buffer = buffer
        self.trainer = trainer
        self.engine = engine
        self.evaluator = evaluator
        self.curator = curator
        self.rehearsal = rehearsal
        self.min_pairs = min_pairs
        self.destroy_rejected = destroy_rejected
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.history: list[dict[str, Any]] = []
        # Observability hook: called with each cycle's report dict.
        self.on_report: Callable[[dict[str, Any]], Any] | None = None

    def _report(self, report: dict[str, Any]) -> dict[str, Any]:
        self.history.append(report)
        if self.on_report is not None:
            self.on_report(report)
        return report

    @staticmethod
    def _max_created_at(logs: list[dict[str, Any]]) -> datetime | None:
        stamps = [row["created_at"] for row in logs if row.get("created_at")]
        if not stamps:
            return None
        return max(datetime.fromisoformat(stamp) for stamp in stamps)

    async def _build_pairs(self, logs: list[dict[str, Any]]) -> list[dict[str, str]]:
        if self.curator is not None:
            pairs = self.curator(logs)
            if asyncio.iscoroutine(pairs):
                pairs = await pairs
            return pairs
        system_prompt = getattr(self.engine, "system_prompt", None)
        return format_training_pairs(logs, system_prompt=system_prompt)

    async def _shadow_gate(self, name: str, path: str) -> tuple[bool, Any]:
        """Load the adapter invisibly, score it, and keep or destroy it."""
        assert self.evaluator is not None  # only called when an evaluator is set
        await self.engine.load_lora_adapter(name, path, activate=False)
        verdict = self.evaluator(name, path)
        if asyncio.iscoroutine(verdict):
            verdict = await verdict
        accepted = bool(getattr(verdict, "accepted", verdict))
        if accepted:
            activate = getattr(self.engine, "activate_lora", None)
            if activate is not None:
                activate(name)
            else:
                await self.engine.load_lora_adapter(name, path, activate=True)
            return True, verdict
        logger.warning("adapter %s rejected by shadow gate: %s", name, verdict)
        unload = getattr(self.engine, "unload_lora_adapter", None)
        if unload is not None:
            await unload(name)
        if self.destroy_rejected:
            shutil.rmtree(path, ignore_errors=True)
        return False, verdict

    async def run_cycle(self) -> dict[str, Any]:
        """One full learning cycle; returns a status report."""
        async with self._lock:
            with traced_span("memory.learning_cycle") as span:
                watermark = await self.buffer.get_watermark()
                logs = await self.buffer.fetch(since=watermark)
                pairs = await self._build_pairs(logs)
                if len(pairs) < self.min_pairs:
                    return self._report(
                        {"status": "skipped", "reason": "not_enough_pairs", "pairs": len(pairs)}
                    )
                new_pairs = len(pairs)
                if self.rehearsal is not None:
                    pairs = self.rehearsal.mix(pairs)

                name, path = await self.trainer.train(pairs)

                if self.evaluator is not None and self.engine is not None:
                    accepted, verdict = await self._shadow_gate(name, path)
                    if not accepted:
                        # Data is not consumed on rejection: it gets another
                        # chance next cycle (possibly combined with new data
                        # or a different rehearsal mix).
                        report = {
                            "status": "rejected",
                            "adapter": name,
                            "reason": "failed_eval_gate",
                            "verdict": getattr(verdict, "__dict__", verdict),
                        }
                        return self._report(report)
                elif self.engine is not None:
                    await self.engine.load_lora_adapter(name, path)

                new_watermark = self._max_created_at(logs)
                if new_watermark is not None:
                    await self.buffer.set_watermark(new_watermark)

                span.set_attributes({"adapter": name, "pairs": len(pairs)})
                report = {
                    "status": "deployed",
                    "adapter": name,
                    "path": path,
                    "pairs": len(pairs),
                    "new_pairs": new_pairs,
                }
                logger.info("deployed adapter %s (%d pairs, %d new)", name, len(pairs), new_pairs)
                return self._report(report)

    def trigger(self) -> asyncio.Task[dict[str, Any]]:
        """Fire-and-forget cycle; suitable as an on_threshold callback.

        The task is retained until done (no mid-flight garbage collection)
        and failures are logged instead of vanishing silently.
        """
        task = asyncio.get_running_loop().create_task(self.run_cycle())
        self._tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error("learning cycle failed", exc_info=t.exception())

        task.add_done_callback(_done)
        return task
