# omniai.telemetry

## `traced_span`

```python
with traced_span("my.operation", {"key": "value"}) as span:
    span.set_attribute("tokens", 42)
    span.set_attributes({...})
```

Context manager instrumenting a block: captures duration, attributes, and errors. If the OpenTelemetry SDK is installed and configured (see `setup_tracing` / `OMNIAI_OTLP_ENDPOINT`), data mirrors into a real OTel span; either way a `SpanRecord` lands in the in-process recorder.

Built-in instrumentation points: `gateway.dispatch`, `graph.node.<name>`, `engine.chat` (token counts), `engine.load_lora`, `memory.lora_train`, `memory.learning_cycle`.

## `SpanRecord` (dataclass)

`name`, `attributes`, `started_at`, `duration_ms`, `error` (`"ExcType: msg"` or `None`).

## `recorder`

Process-global `TelemetryRecorder` with `spans: list[SpanRecord]` and `clear()` — useful in tests:

```python
from omniai.telemetry import recorder
recorder.clear()
...  # exercise code
assert any(s.name == "engine.chat" for s in recorder.spans)
```
