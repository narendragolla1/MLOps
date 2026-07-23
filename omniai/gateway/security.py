"""Gateway security: API-key auth, rate limiting, and body-size caps.

Auth is fail-closed: with auth enabled and no keys configured the app
refuses to start (see ``OmniSettings.validate_security``). Keys travel in
the ``X-API-Key`` header; WebSocket clients may fall back to an ``api_key``
query parameter since browsers cannot set custom WS headers.
"""

from __future__ import annotations

import secrets
import time

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket

from omniai.settings import OmniSettings

WS_POLICY_VIOLATION = 4401  # app-level close code for failed WS auth

EXEMPT_PATHS = {"/health", "/health/live", "/health/ready", "/metrics"}


def _key_valid(candidate: str | None, keys: list[str]) -> bool:
    if not candidate:
        return False
    # compare_digest over every key: constant-time, no early exit.
    return any(secrets.compare_digest(candidate, key) for key in keys)


class TokenBucketRateLimiter:
    """In-process token bucket per API key.

    Interface is deliberately minimal (``allow(key) -> retry_after | None``)
    so a Redis-backed implementation can drop in for multi-replica setups.
    """

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)

    def allow(self, key: str) -> float | None:
        """None if allowed; otherwise seconds to wait before retrying."""
        now = time.monotonic()
        tokens, ts = self._buckets.get(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - ts) * self.rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return None
        self._buckets[key] = (tokens, now)
        return (1.0 - tokens) / self.rate


class SecurityMiddleware:
    """ASGI middleware enforcing auth, rate limits, and body size."""

    def __init__(
        self,
        app: ASGIApp,
        settings: OmniSettings,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ):
        self.app = app
        self.settings = settings
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(
            settings.rate_limit_rps, settings.rate_limit_burst
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        api_key = self._extract_key(scope)
        if self.settings.auth_enabled and not _key_valid(api_key, self.settings.api_keys):
            await self._reject_auth(scope, receive, send)
            return

        retry_after = self.rate_limiter.allow(api_key or "anonymous")
        if retry_after is not None:
            await self._reject_rate(scope, receive, send, retry_after)
            return

        if scope["type"] == "http":
            length = self._content_length(scope)
            if length is not None and length > self.settings.max_body_bytes:
                response = JSONResponse(
                    {"error": {"type": "payload_too_large", "detail": "request body too large"}},
                    status_code=413,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)

    @staticmethod
    def _extract_key(scope: Scope) -> str | None:
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if key := headers.get("x-api-key"):
            return key
        if scope["type"] == "websocket":
            from urllib.parse import parse_qs

            params = parse_qs(scope.get("query_string", b"").decode())
            if values := params.get("api_key"):
                return values[0]
        return None

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def _reject_auth(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            ws = WebSocket(scope, receive=receive, send=send)
            await ws.close(code=WS_POLICY_VIOLATION, reason="invalid or missing API key")
            return
        response = JSONResponse(
            {"error": {"type": "unauthorized", "detail": "invalid or missing API key"}},
            status_code=401,
            headers={"WWW-Authenticate": "ApiKey"},
        )
        await response(scope, receive, send)

    async def _reject_rate(
        self, scope: Scope, receive: Receive, send: Send, retry_after: float
    ) -> None:
        if scope["type"] == "websocket":
            ws = WebSocket(scope, receive=receive, send=send)
            await ws.close(code=WS_POLICY_VIOLATION, reason="rate limit exceeded")
            return
        response = JSONResponse(
            {"error": {"type": "rate_limited", "detail": "rate limit exceeded"}},
            status_code=429,
            headers={"Retry-After": str(max(1, round(retry_after)))},
        )
        await response(scope, receive, send)


def request_api_key(request: Request) -> str | None:
    return request.headers.get("x-api-key")
