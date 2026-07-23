"""GatewayRouter: the multi-channel control plane.

Wraps a FastAPI app with REST, WebSocket, and Discord routes. Every route
follows the same pipeline:

    native payload -> adapter.to_omni -> guardrails -> handler (graph)
                   -> interaction logging -> adapter.from_omni -> response
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from omniai.gateway.adapters import DiscordAdapter, RestAdapter, WebSocketAdapter
from omniai.protocol import OmniMessage
from omniai.settings import OmniSettings
from omniai.telemetry import traced_span

Handler = Callable[[OmniMessage], OmniMessage | Awaitable[OmniMessage]]
Interceptor = Callable[[OmniMessage], OmniMessage | Awaitable[OmniMessage]]
Observer = Callable[[OmniMessage], Any]


class GuardrailViolation(Exception):
    """Raised by an interceptor to reject a message before it reaches the graph."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class GatewayRouter:
    """Central router owning the FastAPI app and the message pipeline.

    Parameters
    ----------
    handler:
        Callable (sync or async) mapping an inbound OmniMessage to a reply —
        typically ``compiled_graph.as_handler()``.
    interceptors:
        Middleware run on inbound messages before the handler (guardrails).
        May mutate/replace the message or raise :class:`GuardrailViolation`.
    observers:
        Fire-and-forget callbacks (sync or async) invoked with every inbound
        and outbound message — e.g. the memory InteractionBuffer.
    settings:
        Production hardening switch. When provided, the router validates the
        security config (fail-closed on missing API keys), and installs
        auth + rate-limit + body-size middleware and CORS. When omitted the
        router runs open — embedded/test mode only.
    """

    def __init__(
        self,
        handler: Handler,
        interceptors: list[Interceptor] | None = None,
        observers: list[Observer] | None = None,
        app: FastAPI | None = None,
        settings: OmniSettings | None = None,
        shutdown_hooks: list[Callable[[], Any]] | None = None,
        engine: Any = None,
        buffer: Any = None,
    ):
        self.handler = handler
        self.interceptors = list(interceptors or [])
        self.observers = list(observers or [])
        self.app = app or FastAPI(title="OmniAI Gateway")
        self.settings = settings
        self.engine = engine
        self.buffer = buffer
        self.metrics: Any = None
        self.rest = RestAdapter()
        self.ws = WebSocketAdapter()
        self.discord = DiscordAdapter()
        self._register_routes()
        self._register_error_handlers()
        for hook in shutdown_hooks or []:
            self.app.router.add_event_handler("shutdown", hook)
        if settings is not None:
            # First added = innermost: the body limit must sit inside the
            # observability BaseHTTPMiddlewares so its 413 surfaces through
            # FastAPI's exception handling (see BodyLimitMiddleware).
            from omniai.gateway.security import BodyLimitMiddleware

            self.app.add_middleware(BodyLimitMiddleware, settings=settings)
            self._apply_observability(settings)
            self._apply_security(settings)

    def _apply_observability(self, settings: OmniSettings) -> None:
        from omniai.engine.resilience import BreakerState
        from omniai.gateway.observability import (
            Metrics,
            configure_logging,
            metrics_middleware,
            request_id_middleware,
            setup_tracing,
        )

        configure_logging(settings)
        setup_tracing(settings)
        metrics = self.metrics = Metrics()
        # Middleware runs outermost-last-added: metrics inside request-id.
        self.app.middleware("http")(metrics_middleware(metrics))
        self.app.middleware("http")(request_id_middleware())

        if self.engine is not None:

            def _record_usage(prompt: int, completion: int) -> None:
                metrics.tokens.labels("prompt").inc(prompt)
                metrics.tokens.labels("completion").inc(completion)

            self.engine.on_usage = _record_usage

        @self.app.get("/metrics")
        async def metrics_endpoint() -> Any:
            from fastapi.responses import Response

            if self.engine is not None:
                metrics.breaker_state.set(
                    1 if self.engine.breaker.state is BreakerState.OPEN else 0
                )
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)

        @self.app.get("/health/live")
        async def liveness() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.get("/health/ready")
        async def readiness() -> Any:
            from fastapi.responses import JSONResponse

            problems: list[str] = []
            if self.buffer is not None:
                try:
                    await self.buffer.count()
                except Exception as exc:
                    problems.append(f"database: {type(exc).__name__}")
            if self.engine is not None:
                if self.engine.breaker.state is BreakerState.OPEN:
                    problems.append("engine: circuit breaker open")
                supervisor = self.engine.supervisor
                if supervisor is not None and supervisor.failed:
                    problems.append(f"engine: supervisor failed ({supervisor.failure_reason})")
            if problems:
                return JSONResponse(
                    status_code=503, content={"status": "unready", "problems": problems}
                )
            return {"status": "ready"}

    def _register_error_handlers(self) -> None:
        """Problem-details JSON for failures; never leak stack traces."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        from omniai.engine.resilience import EngineUnavailable
        from omniai.gateway.security import BodyLimitExceeded

        @self.app.exception_handler(EngineUnavailable)
        async def engine_unavailable(request: Request, exc: EngineUnavailable) -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={"error": {"type": "engine_unavailable", "detail": str(exc)}},
                headers={"Retry-After": "5"},
            )

        @self.app.exception_handler(BodyLimitExceeded)
        async def body_too_large(request: Request, exc: BodyLimitExceeded) -> JSONResponse:
            return JSONResponse(
                status_code=413,
                content={"error": {"type": "payload_too_large", "detail": str(exc)}},
            )

        @self.app.exception_handler(Exception)
        async def unhandled(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=500,
                content={"error": {"type": "internal_error", "detail": "internal server error"}},
            )

    def _apply_security(self, settings: OmniSettings) -> None:
        from starlette.middleware.cors import CORSMiddleware

        from omniai.gateway.security import SecurityMiddleware

        settings.validate_security()
        if settings.cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=settings.cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        self.app.add_middleware(SecurityMiddleware, settings=settings)

    def add_interceptor(self, interceptor: Interceptor) -> None:
        self.interceptors.append(interceptor)

    def add_observer(self, observer: Observer) -> None:
        self.observers.append(observer)

    async def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        result = fn(*args)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _notify(self, message: OmniMessage) -> None:
        for observer in self.observers:
            await self._call(observer, message)

    async def dispatch(self, message: OmniMessage) -> OmniMessage:
        """Run one message through interceptors, handler, and observers."""
        with traced_span(
            "gateway.dispatch", {"channel": message.channel.value, "session": message.session_id}
        ):
            for interceptor in self.interceptors:
                message = await self._call(interceptor, message)
            await self._notify(message)
            reply = await self._call(self.handler, message)
            await self._notify(reply)
            return reply

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/v1/messages")
        async def rest_message(payload: dict[str, Any]) -> dict[str, Any]:
            message = self.rest.to_omni(payload)
            try:
                reply = await self.dispatch(message)
            except GuardrailViolation as exc:
                raise HTTPException(status_code=400, detail=exc.reason) from exc
            return self.rest.from_omni(reply)

        @app.post("/discord/webhook")
        async def discord_webhook(payload: dict[str, Any]) -> dict[str, Any]:
            message = self.discord.to_omni(payload)
            try:
                reply = await self.dispatch(message)
            except GuardrailViolation as exc:
                raise HTTPException(status_code=400, detail=exc.reason) from exc
            return self.discord.from_omni(reply)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            try:
                while True:
                    payload = await websocket.receive_json()
                    message = self.ws.to_omni(payload)
                    try:
                        reply = await self.dispatch(message)
                    except GuardrailViolation as exc:
                        await websocket.send_json({"error": exc.reason})
                        continue
                    await websocket.send_json(self.ws.from_omni(reply))
            except WebSocketDisconnect:
                pass
