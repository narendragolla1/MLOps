"""OpenTelemetry instrumentation with a zero-dependency fallback.

If ``opentelemetry`` is installed, spans are exported through the configured
tracer provider; otherwise a lightweight recorder keeps span data in-process
so tests and local runs can still inspect latency and token counts.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised only when otel is installed
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("omniai")
except ImportError:
    _TRACER = None


@dataclass
class SpanRecord:
    """In-process record of one instrumented operation."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)


class TelemetryRecorder:
    """Collects finished spans; the sink for the no-op tracer."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []

    def record(self, span: SpanRecord) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()


recorder = TelemetryRecorder()


@contextlib.contextmanager
def traced_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[SpanRecord]:
    """Instrument a block: captures duration, attributes, and errors.

    Always yields a :class:`SpanRecord`; mirrors data into a real OTel span
    when the SDK is available.
    """
    record = SpanRecord(name=name, attributes=dict(attributes or {}), started_at=time.time())
    otel_cm = (
        _TRACER.start_as_current_span(name) if _TRACER is not None else contextlib.nullcontext()
    )
    with otel_cm as otel_span:
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.duration_ms = (time.time() - record.started_at) * 1000
            if otel_span is not None and hasattr(otel_span, "set_attribute"):
                for key, value in record.attributes.items():
                    with contextlib.suppress(Exception):
                        otel_span.set_attribute(key, value)
            recorder.record(record)


__all__ = ["traced_span", "recorder", "SpanRecord", "TelemetryRecorder"]
