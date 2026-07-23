"""GatewayRouter: the multi-channel control plane.

Wraps a FastAPI app with REST, WebSocket, and Discord routes. Every route
follows the same pipeline:

    native payload -> adapter.to_omni -> guardrails -> handler (graph)
                   -> interaction logging -> adapter.from_omni -> response
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from omniai.gateway.adapters import DiscordAdapter, RestAdapter, WebSocketAdapter
from omniai.protocol import OmniMessage
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
    """

    def __init__(
        self,
        handler: Handler,
        interceptors: list[Interceptor] | None = None,
        observers: list[Observer] | None = None,
        app: FastAPI | None = None,
    ):
        self.handler = handler
        self.interceptors = list(interceptors or [])
        self.observers = list(observers or [])
        self.app = app or FastAPI(title="OmniAI Gateway")
        self.rest = RestAdapter()
        self.ws = WebSocketAdapter()
        self.discord = DiscordAdapter()
        self._register_routes()

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
