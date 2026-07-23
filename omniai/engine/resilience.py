"""Reliability primitives for engine HTTP calls and subprocess lifecycle.

- :func:`with_retries` — exponential backoff + jitter on transient failures.
- :class:`CircuitBreaker` — fail fast while the backend is down, probe on a
  timer (closed -> open -> half-open).
- :class:`EngineSupervisor` — watches the backend subprocess, restarts it
  with capped backoff, and re-applies the active LoRA adapter afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

import httpx

T = TypeVar("T")


class EngineUnavailable(Exception):
    """The serving backend is down or the circuit breaker is open."""


def _is_transient(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 4.0,
) -> T:
    """Run ``fn``, retrying transient failures with backoff + full jitter."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2**attempt)) * random.random()
            await asyncio.sleep(delay)
    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc  # pragma: no cover


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker around an async operation."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at = 0.0

    def _maybe_half_open(self) -> None:
        if (
            self.state is BreakerState.OPEN
            and time.monotonic() - self.opened_at >= self.reset_timeout
        ):
            self.state = BreakerState.HALF_OPEN

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        self._maybe_half_open()
        if self.state is BreakerState.OPEN:
            raise EngineUnavailable(f"circuit breaker open (retry in <= {self.reset_timeout}s)")
        try:
            result = await fn()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _record_failure(self) -> None:
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = time.monotonic()

    def _record_success(self) -> None:
        self.failures = 0
        self.state = BreakerState.CLOSED


class EngineSupervisor:
    """Restarts a crashed backend subprocess and restores the active LoRA."""

    def __init__(
        self,
        engine: Any,  # ModelEngine; Any avoids a circular import
        check_interval: float = 2.0,
        max_restarts: int = 5,
        ready_timeout: float = 300.0,
        restart_backoff_base: float = 0.5,
    ):
        self.engine = engine
        self.check_interval = check_interval
        self.max_restarts = max_restarts
        self.ready_timeout = ready_timeout
        self.restart_backoff_base = restart_backoff_base
        self.restarts = 0
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stopped.clear()
            self._task = asyncio.get_running_loop().create_task(self._watch())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def _process_alive(self) -> bool:
        process = self.engine.adapter.process
        return process is not None and process.poll() is None

    async def _watch(self) -> None:
        while not self._stopped.is_set():
            await asyncio.sleep(self.check_interval)
            if self._stopped.is_set() or self._process_alive():
                continue
            if self.restarts >= self.max_restarts:
                raise EngineUnavailable(
                    f"backend crashed and exceeded max_restarts={self.max_restarts}"
                )
            await self._restart()

    async def _restart(self) -> None:
        self.restarts += 1
        # Capped exponential backoff between restart attempts.
        await asyncio.sleep(min(30.0, self.restart_backoff_base * (2**self.restarts)))
        self.engine.adapter.start()
        ready = await self.engine.adapter.wait_ready(timeout=self.ready_timeout)
        if not ready:
            return  # next watch iteration sees the dead process and retries
        # A fresh server has lost dynamically loaded adapters: re-apply.
        if self.engine.active_lora and self.engine.active_lora_path:
            await self.engine.load_lora_adapter(
                self.engine.active_lora, self.engine.active_lora_path
            )
