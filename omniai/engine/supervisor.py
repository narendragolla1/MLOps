"""EngineSupervisor: keep a self-hosted backend alive without a human.

A background watchdog polling the engine's process and HTTP health. On
failure it restarts the backend with exponential crash-loop backoff, waits
for readiness, re-applies the active LoRA adapter from the registry, and
warms the server back up. After ``max_restarts`` consecutive failed
recoveries it stops trying (circuit open) instead of thrashing the GPU, and
reports everything through an injectable ``on_event`` callback.

Usage::

    supervisor = EngineSupervisor(engine, on_event=print)
    await supervisor.start()      # returns immediately; watchdog runs in background
    ...
    await supervisor.stop()
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from omniai.telemetry import traced_span

EventHook = Callable[[str, dict[str, Any]], Any]


class EngineSupervisor:
    """Watchdog + restart policy for a ModelEngine's backend process."""

    def __init__(
        self,
        engine,
        check_interval: float = 5.0,
        max_restarts: int = 3,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        ready_timeout: float = 300.0,
        on_event: EventHook | None = None,
        probe: Callable[[], Awaitable[bool]] | None = None,
    ):
        self.engine = engine
        self.check_interval = check_interval
        self.max_restarts = max_restarts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.ready_timeout = ready_timeout
        self.on_event = on_event
        self._probe = probe  # test seam: async () -> bool (healthy?)
        self._task: asyncio.Task | None = None
        self._failures = 0  # consecutive failed recoveries
        self.restarts = 0  # total successful restarts

    @property
    def gave_up(self) -> bool:
        return self._failures >= self.max_restarts

    def _emit(self, event: str, **data: Any) -> None:
        if self.on_event is not None:
            self.on_event(event, data)

    async def _healthy(self) -> bool:
        if self._probe is not None:
            return await self._probe()
        if not self.engine.adapter.is_alive():
            return False
        status = await self.engine.health()
        return bool(status["server"])

    async def _recover(self) -> bool:
        with traced_span("engine.supervisor.restart", {"attempt": self._failures + 1}):
            self.engine.adapter.stop()
            self.engine.adapter.start()
            ready = await self.engine.adapter.wait_ready(timeout=self.ready_timeout)
            if not ready:
                return False
            try:
                await self.engine.reapply_active_lora()
            except Exception as exc:
                self._emit("lora_reapply_failed", error=str(exc))
            await self.engine.warmup()
            return True

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self.check_interval)
            if await self._healthy():
                self._failures = 0
                continue
            self._emit("unhealthy", failures=self._failures)
            if await self._recover():
                self.restarts += 1
                self._failures = 0
                self._emit("restarted", restarts=self.restarts)
                continue
            self._failures += 1
            if self.gave_up:
                self._emit("gave_up", failures=self._failures)
                return
            delay = min(self.backoff_max, self.backoff_base * 2 ** (self._failures - 1))
            self._emit("backoff", delay=delay, failures=self._failures)
            await asyncio.sleep(delay)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("Supervisor already running")
        self._failures = 0
        self._task = asyncio.create_task(self._watch())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
