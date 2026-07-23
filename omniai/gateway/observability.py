"""Gateway observability: metrics, request IDs, structured logs, probes.

Everything here is wired by ``GatewayRouter._apply_observability``:
  - request-ID middleware (honors inbound ``X-Request-ID``, echoes it back,
    exposes it to logs via a contextvar),
  - Prometheus metrics middleware + ``/metrics`` endpoint,
  - ``/health/live`` and ``/health/ready`` probes,
  - JSON logging setup for the process.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

from omniai.settings import OmniSettings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class Metrics:
    """Prometheus instruments; one registry per router (test-friendly)."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "omniai_requests_total",
            "HTTP requests processed",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "omniai_request_latency_seconds",
            "HTTP request latency",
            ["method", "path"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "omniai_engine_tokens_total",
            "Tokens processed by the engine",
            ["kind"],  # prompt | completion
            registry=self.registry,
        )
        self.breaker_state = Gauge(
            "omniai_breaker_open",
            "1 when the engine circuit breaker is open",
            registry=self.registry,
        )
        self.learning_cycles = Counter(
            "omniai_learning_cycles_total",
            "Continuous-learning cycles by outcome",
            ["status"],
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class JsonFormatter(logging.Formatter):
    """Single-line JSON log records with the current request ID."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str)


def configure_logging(settings: OmniSettings) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]


def request_id_middleware() -> Callable[[Request, Callable], Awaitable[Response]]:
    async def middleware(request: Request, call_next: Callable) -> Response:
        rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        token = request_id_var.set(rid)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    return middleware


def metrics_middleware(metrics: Metrics) -> Callable[[Request, Callable], Awaitable[Response]]:
    async def middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        # Label with the matched route template, never the raw URL: raw
        # paths (e.g. scanner 404s) would grow Prometheus series without
        # bound. Unrouted requests share one "unmatched" label.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.requests.labels(request.method, path, str(response.status_code)).inc()
        metrics.latency.labels(request.method, path).observe(elapsed)
        return response

    return middleware


def setup_tracing(settings: OmniSettings) -> bool:
    """Wire the OTLP exporter when configured and the SDK is installed."""
    if not settings.otlp_endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.getLogger(__name__).warning(
            "OMNIAI_OTLP_ENDPOINT set but opentelemetry SDK not installed; "
            "install omniai[telemetry]"
        )
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    return True
