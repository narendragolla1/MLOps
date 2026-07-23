"""ModelEngine: factory facade over the serving backends.

The engine owns the backend subprocess lifecycle and exposes a single async
OpenAI-compatible client interface (``chat``) plus a managed LoRA lifecycle
(load / unload / activate / rollback via :class:`~omniai.engine.lora.LoRARegistry`),
so the rest of the framework never touches backend-specific details.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from omniai.engine.backends import ADAPTERS, BackendAdapter
from omniai.engine.config import EngineConfig
from omniai.engine.lora import LoRARegistry
from omniai.engine.resilience import (
    CircuitBreaker,
    EngineSupervisor,
    EngineUnavailable,
    with_retries,
)
from omniai.telemetry import traced_span


class ModelEngine:
    """Unified serving facade. Create via :meth:`ModelEngine.create`."""

    def __init__(
        self,
        config: EngineConfig,
        adapter: BackendAdapter,
        lora_registry: LoRARegistry | None = None,
    ):
        self.config = config
        self.adapter = adapter
        self.system_prompt: str | None = None
        self.lora = lora_registry or LoRARegistry()
        self.breaker = CircuitBreaker(
            failure_threshold=config.breaker_failure_threshold,
            reset_timeout=config.breaker_reset_s,
        )
        self.supervisor: EngineSupervisor | None = None
        # Observability hook: called with (prompt_tokens, completion_tokens).
        self.on_usage: Any = None
        self.in_flight = 0
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        # Created lazily so the engine can be constructed outside a loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        return self._semaphore

    @classmethod
    def create(cls, config: EngineConfig | dict[str, Any], **kwargs: Any) -> ModelEngine:
        """Factory: build the engine with the adapter for ``config.backend``."""
        if isinstance(config, dict):
            config = EngineConfig(**config)
        adapter_cls = ADAPTERS.get(config.backend_name)
        if adapter_cls is None:
            raise ValueError(f"Unknown backend: {config.backend}")
        return cls(config, adapter_cls(config), **kwargs)

    @property
    def active_lora(self) -> str | None:
        return self.lora.active

    @property
    def active_lora_path(self) -> str | None:
        """Path of the active adapter, or None on the base model.

        Kept alongside :attr:`active_lora` for the supervisor's restart path,
        which re-applies both without reaching into the registry directly.
        """
        if self.lora.active is None:
            return None
        return self.lora.loaded[self.lora.active].path

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url, timeout=self.config.request_timeout_s
            )
        return self._client

    async def start(
        self, wait: bool = True, timeout: float = 300.0, supervise: bool = False
    ) -> None:
        """Launch (or attach to) the backend; optionally supervise it.

        In unmanaged mode (``config.managed=False``) no subprocess is
        spawned — the engine attaches to the server at
        ``config.external_base_url`` and only health-checks it.
        """
        if not self.config.managed:
            if wait and not await self.adapter.wait_ready(timeout=timeout):
                raise EngineUnavailable(
                    f"external engine at {self.config.base_url} not ready within {timeout}s"
                )
            return
        self.adapter.start()
        if wait:
            ready = await self.adapter.wait_ready(timeout=timeout)
            if not ready:
                self.adapter.stop()
                raise EngineUnavailable(
                    f"{self.config.backend_name} server did not become ready within {timeout}s"
                )
        if supervise:
            self.supervisor = EngineSupervisor(self)
            self.supervisor.start()

    async def stop(self) -> None:
        if self.supervisor is not None:
            await self.supervisor.stop()
            self.supervisor = None
        if self.config.managed:
            self.adapter.stop()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> dict[str, Any]:
        """Liveness of the managed process and the HTTP endpoint."""
        server_ok = False
        try:
            resp = await self.client.get("/health")
            server_ok = resp.status_code == 200
        except httpx.HTTPError:
            pass
        return {
            "process": self.adapter.is_alive(),
            "server": server_ok,
            "active_lora": self.active_lora,
        }

    async def warmup(self) -> bool:
        """One tiny generation so CUDA graphs/caches are primed before real
        traffic; returns False instead of raising on failure."""
        try:
            await self.chat_text([{"role": "user", "content": "ping"}], max_tokens=1)
            return True
        except (httpx.HTTPError, KeyError):
            return False

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with retry + circuit breaker; raises EngineUnavailable when down."""

        async def attempt() -> httpx.Response:
            resp = await self.client.post(path, json=payload)
            resp.raise_for_status()
            return resp

        async def guarded() -> httpx.Response:
            return await with_retries(attempt, attempts=self.config.retries)

        try:
            async with self.semaphore:  # backpressure toward the backend
                self.in_flight += 1
                try:
                    return await self.breaker.call(guarded)
                finally:
                    self.in_flight -= 1
        except EngineUnavailable:
            raise
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise  # client errors are the caller's bug, not availability
            raise EngineUnavailable(f"engine request failed: {exc}") from exc

    def set_system_prompt(self, prompt: str) -> None:
        """Pre-cache a system prompt prepended to every conversation.

        With SGLang this shared prefix is served from the RadixAttention
        cache, so long skill prompts cost prefill only once.
        """
        self.system_prompt = prompt

    def _build_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            return [{"role": "system", "content": self.system_prompt}, *messages]
        return list(messages)

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion against the live backend."""
        payload: dict[str, Any] = {
            "model": self.active_lora or self.config.model,
            "messages": self._build_messages(messages),
            **kwargs,
        }
        with traced_span(
            "engine.chat", {"model": payload["model"], "backend": self.config.backend_name}
        ) as span:
            resp = await self._post("/v1/chat/completions", payload)
            data = resp.json()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            span.set_attributes(
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
            )
            if self.on_usage is not None:
                self.on_usage(prompt_tokens, completion_tokens)
            return data

    async def chat_text(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Convenience wrapper returning just the first choice's content."""
        data = await self.chat(messages, **kwargs)
        return data["choices"][0]["message"]["content"]

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream completion deltas as they arrive (SSE)."""
        payload: dict[str, Any] = {
            "model": self.active_lora or self.config.model,
            "messages": self._build_messages(messages),
            "stream": True,
            **kwargs,
        }
        async with self.client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line.removeprefix("data:").strip()
                if chunk == "[DONE]":
                    break
                import json

                delta = json.loads(chunk)["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content

    # -- LoRA lifecycle ------------------------------------------------------

    async def load_lora_adapter(self, name: str, path: str, activate: bool = True) -> bool:
        """Load an adapter into the running server, zero downtime.

        When the server's adapter slots (``config.max_loras``) are full, the
        oldest loaded adapter that is neither active nor the rollback target
        is evicted first, so the continuous-learning loop never stalls on a
        full server.
        """
        victim = self.lora.eviction_candidate(self.config.max_loras)
        if victim is not None:
            await self.unload_lora_adapter(victim)
        with traced_span("engine.load_lora", {"adapter": name}):
            await self._post(
                self.adapter.lora_load_endpoint(), self.adapter.lora_load_payload(name, path)
            )
        self.lora.register(name, path)
        if activate:
            self.lora.activate(name)
        return True

    async def unload_lora_adapter(self, name: str) -> bool:
        """Remove an adapter from the server and the registry."""
        with traced_span("engine.unload_lora", {"adapter": name}):
            await self._post(
                self.adapter.lora_unload_endpoint(), self.adapter.lora_unload_payload(name)
            )
        self.lora.remove(name)
        return True

    async def rollback_lora(self) -> str | None:
        """Reactivate the previously active adapter (or the base model).

        Returns the adapter now active, or None when back on the base model.
        The rolled-back-from adapter stays loaded, so rolling forward again
        is equally cheap.
        """
        target = self.lora.previous
        if target is None:
            self.lora.deactivate()
            return None
        self.lora.activate(target)
        return target

    async def reapply_active_lora(self) -> bool:
        """Re-load the registry's active adapter into a restarted server."""
        if self.lora.active is None:
            return False
        record = self.lora.loaded[self.lora.active]
        await self._post(
            self.adapter.lora_load_endpoint(),
            self.adapter.lora_load_payload(record.name, record.path),
        )
        return True
