"""Continuous learning: interaction logs -> LoRA adapter -> zero-downtime swap.

Pipeline (see :class:`ContinuousLearner.run_cycle`):
  1. pull interaction logs from the :class:`InteractionBuffer`,
  2. format them into instruction-tuning pairs,
  3. train a PEFT LoRA adapter off the event loop (thread or subprocess),
  4. optionally gate the adapter through an evaluator,
  5. hot-swap the adapter into the live engine via its REST API.

The heavy Hugging Face training runs only when ``peft``/``trl`` are
installed; a custom ``train_fn`` can be injected for tests or bespoke loops.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

from omniai.telemetry import traced_span


def format_training_pairs(
    logs: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Convert interaction logs into instruction-tuning pairs.

    Pairs each user message with the next assistant message in the same
    session; tool outputs in between are folded into the prompt as context.
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
                pairs.append(pair)
                prompt, context = None, []
    return pairs


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
        self._version = 0

    async def train(self, pairs: list[dict[str, str]]) -> tuple[str, str]:
        """Train an adapter; returns (adapter_name, adapter_path)."""
        if not pairs:
            raise ValueError("No training pairs; refusing to train an empty adapter")
        self._version += 1
        name = f"{Path(self.base_model).name}-lora-v{self._version}"
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
    """Orchestrates the log -> train -> evaluate -> hot-swap cycle.

    Wire ``learner.trigger`` as the buffer's ``on_threshold`` callback for
    automatic cycles, or call ``await learner.run_cycle()`` from an admin
    endpoint for manual triggering. Cycles are serialized by a lock so
    overlapping triggers cannot start concurrent trainings.
    """

    def __init__(
        self,
        buffer,
        trainer: LoRATrainer,
        engine=None,
        evaluator: Callable[[str, str], Any] | None = None,
        min_pairs: int = 1,
    ):
        self.buffer = buffer
        self.trainer = trainer
        self.engine = engine
        self.evaluator = evaluator
        self.min_pairs = min_pairs
        self._lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []
        # Observability hook: called with each cycle's report dict.
        self.on_report: Callable[[dict[str, Any]], Any] | None = None

    def _report(self, report: dict[str, Any]) -> dict[str, Any]:
        self.history.append(report)
        if self.on_report is not None:
            self.on_report(report)
        return report

    async def run_cycle(self) -> dict[str, Any]:
        """One full learning cycle; returns a status report."""
        async with self._lock:
            with traced_span("memory.learning_cycle") as span:
                logs = await self.buffer.fetch()
                system_prompt = getattr(self.engine, "system_prompt", None)
                pairs = format_training_pairs(logs, system_prompt=system_prompt)
                if len(pairs) < self.min_pairs:
                    return self._report(
                        {"status": "skipped", "reason": "not_enough_pairs", "pairs": len(pairs)}
                    )

                name, path = await self.trainer.train(pairs)

                if self.evaluator is not None:
                    verdict = self.evaluator(name, path)
                    if asyncio.iscoroutine(verdict):
                        verdict = await verdict
                    if not getattr(verdict, "accepted", verdict):
                        report = {
                            "status": "rejected",
                            "adapter": name,
                            "reason": "failed_eval_gate",
                            "verdict": getattr(verdict, "__dict__", verdict),
                        }
                        return self._report(report)

                if self.engine is not None:
                    await self.engine.load_lora_adapter(name, path)

                span.set_attributes({"adapter": name, "pairs": len(pairs)})
                return self._report(
                    {"status": "deployed", "adapter": name, "path": path, "pairs": len(pairs)}
                )

    def trigger(self) -> "asyncio.Task[dict[str, Any]]":
        """Fire-and-forget cycle; suitable as an on_threshold callback."""
        return asyncio.get_running_loop().create_task(self.run_cycle())
