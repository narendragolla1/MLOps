import asyncio
import time
from unittest import mock

import pytest

from omniai.telemetry import SpanRecord, TelemetryRecorder, traced_span, recorder


# -- SpanRecord tests -------------------------------------------------------


def test_span_record_creation():
    """Verify SpanRecord initializes with correct defaults."""
    record = SpanRecord(name="test_op")
    assert record.name == "test_op"
    assert record.attributes == {}
    assert record.started_at == 0.0
    assert record.duration_ms == 0.0
    assert record.error is None


def test_span_record_set_attribute():
    """Verify single attribute can be set."""
    record = SpanRecord(name="op")
    record.set_attribute("key", "value")
    assert record.attributes["key"] == "value"


def test_span_record_set_attributes():
    """Verify multiple attributes can be set at once."""
    record = SpanRecord(name="op")
    record.set_attributes({"a": 1, "b": "two", "c": [3]})
    assert record.attributes == {"a": 1, "b": "two", "c": [3]}


def test_span_record_set_attributes_overwrites():
    """Verify set_attributes updates existing keys."""
    record = SpanRecord(name="op", attributes={"old": "value"})
    record.set_attributes({"old": "new", "fresh": 42})
    assert record.attributes == {"old": "new", "fresh": 42}


def test_span_record_with_initial_attributes():
    """Verify SpanRecord can be created with initial attributes."""
    attrs = {"key": "value", "count": 123}
    record = SpanRecord(name="op", attributes=attrs)
    assert record.attributes == attrs


# -- TelemetryRecorder tests ------------------------------------------------


def test_telemetry_recorder_starts_empty():
    """Verify recorder initializes with no spans."""
    rec = TelemetryRecorder()
    assert rec.spans == []


def test_telemetry_recorder_records_spans():
    """Verify recorder appends recorded spans."""
    rec = TelemetryRecorder()
    span1 = SpanRecord(name="op1")
    span2 = SpanRecord(name="op2")
    rec.record(span1)
    rec.record(span2)
    assert len(rec.spans) == 2
    assert rec.spans[0].name == "op1"
    assert rec.spans[1].name == "op2"


def test_telemetry_recorder_clears():
    """Verify recorder.clear() empties spans."""
    rec = TelemetryRecorder()
    rec.record(SpanRecord(name="op1"))
    rec.record(SpanRecord(name="op2"))
    assert len(rec.spans) == 2
    rec.clear()
    assert rec.spans == []


# -- traced_span tests: happy path ------------------------------------------


def test_traced_span_basic_recording():
    """Verify traced_span records a span in the global recorder."""
    recorder.clear()
    with traced_span("test_op") as span:
        assert span.name == "test_op"
        assert span.error is None

    assert len(recorder.spans) == 1
    assert recorder.spans[0].name == "test_op"
    assert recorder.spans[0].error is None


def test_traced_span_records_duration():
    """Verify span duration is measured accurately."""
    recorder.clear()
    sleep_time = 0.05
    with traced_span("slow_op"):
        time.sleep(sleep_time)

    assert len(recorder.spans) == 1
    duration_ms = recorder.spans[0].duration_ms
    # Allow some tolerance for timing variance
    assert sleep_time * 1000 <= duration_ms <= (sleep_time * 1000) + 50


def test_traced_span_with_attributes():
    """Verify attributes passed to traced_span are recorded."""
    recorder.clear()
    with traced_span("op", attributes={"user": "alice", "tokens": 123}):
        pass

    assert recorder.spans[0].attributes == {"user": "alice", "tokens": 123}


def test_traced_span_with_set_attribute():
    """Verify attributes can be set within the span."""
    recorder.clear()
    with traced_span("op", attributes={"initial": "value"}) as span:
        span.set_attribute("added", "later")

    attrs = recorder.spans[0].attributes
    assert attrs["initial"] == "value"
    assert attrs["added"] == "later"


def test_traced_span_with_set_attributes():
    """Verify multiple attributes can be set within the span."""
    recorder.clear()
    with traced_span("op") as span:
        span.set_attributes({"key1": "val1", "key2": 42})

    assert recorder.spans[0].attributes == {"key1": "val1", "key2": 42}


def test_traced_span_none_attributes():
    """Verify None attributes defaults to empty dict."""
    recorder.clear()
    with traced_span("op", attributes=None):
        pass

    assert recorder.spans[0].attributes == {}


# -- traced_span tests: error handling --------------------------------------


def test_traced_span_captures_exception():
    """Verify exceptions are recorded in the span."""
    recorder.clear()
    with pytest.raises(ValueError):
        with traced_span("failing_op"):
            raise ValueError("test error")

    assert len(recorder.spans) == 1
    assert recorder.spans[0].error == "ValueError: test error"


def test_traced_span_captures_different_exception_types():
    """Verify different exception types are recorded correctly."""
    test_cases = [
        (KeyError, "missing_key", "KeyError: 'missing_key'"),
        (RuntimeError, "something broke", "RuntimeError: something broke"),
        (TypeError, "bad type", "TypeError: bad type"),
    ]

    for exc_type, exc_msg, expected_error in test_cases:
        recorder.clear()
        with pytest.raises(exc_type):
            with traced_span("op"):
                raise exc_type(exc_msg)
        assert recorder.spans[0].error == expected_error


def test_traced_span_exception_does_not_prevent_recording():
    """Verify span is still recorded even when exception occurs."""
    recorder.clear()
    try:
        with traced_span("op", attributes={"key": "value"}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert len(recorder.spans) == 1
    assert recorder.spans[0].name == "op"
    assert recorder.spans[0].attributes == {"key": "value"}
    assert "RuntimeError" in recorder.spans[0].error


def test_traced_span_exception_still_records_duration():
    """Verify duration is recorded even when exception occurs."""
    recorder.clear()
    sleep_time = 0.03
    try:
        with traced_span("op"):
            time.sleep(sleep_time)
            raise ValueError("error")
    except ValueError:
        pass

    assert recorder.spans[0].duration_ms >= sleep_time * 1000


# -- traced_span tests: async -----------------------------------------------


async def test_traced_span_async_support():
    """Verify traced_span works with async context."""
    recorder.clear()
    with traced_span("async_op", attributes={"async": True}) as span:
        await asyncio.sleep(0.01)
        assert span.name == "async_op"

    assert len(recorder.spans) == 1
    assert recorder.spans[0].attributes["async"] is True


async def test_traced_span_async_exception():
    """Verify exceptions in async context are captured."""
    recorder.clear()
    with pytest.raises(RuntimeError):
        with traced_span("async_failing"):
            await asyncio.sleep(0.001)
            raise RuntimeError("async error")

    assert recorder.spans[0].error == "RuntimeError: async error"


# -- Integration tests: state and side effects ---------------------------


def test_traced_span_records_to_global_recorder():
    """Verify traced_span always records to the global recorder."""
    recorder.clear()
    with traced_span("op1"):
        pass
    with traced_span("op2"):
        pass
    with traced_span("op3"):
        pass

    assert len(recorder.spans) == 3
    assert [s.name for s in recorder.spans] == ["op1", "op2", "op3"]


def test_traced_span_nested_spans():
    """Verify nested spans are all recorded."""
    recorder.clear()
    with traced_span("outer", attributes={"level": "1"}):
        with traced_span("inner", attributes={"level": "2"}):
            pass

    assert len(recorder.spans) == 2
    assert recorder.spans[0].name == "inner"  # inner finishes first
    assert recorder.spans[1].name == "outer"  # outer finishes second


def test_traced_span_nested_with_exception():
    """Verify exception in nested span doesn't affect outer."""
    recorder.clear()
    try:
        with traced_span("outer"):
            with traced_span("inner"):
                raise ValueError("inner failed")
    except ValueError:
        pass

    assert len(recorder.spans) == 2
    inner = next(s for s in recorder.spans if s.name == "inner")
    outer = next(s for s in recorder.spans if s.name == "outer")
    assert "ValueError" in inner.error
    assert outer.error is None


def test_traced_span_multiple_sequential_calls():
    """Verify recorder accumulates spans from multiple calls."""
    recorder.clear()
    for i in range(5):
        with traced_span(f"op_{i}"):
            pass

    assert len(recorder.spans) == 5
    assert [s.name for s in recorder.spans] == [f"op_{i}" for i in range(5)]


# -- OpenTelemetry integration tests ----------------------------------------


def test_traced_span_otel_integration_when_available():
    """Verify attributes are passed to OTel when tracer is available."""
    import omniai.telemetry as tel_module

    if tel_module._TRACER is None:
        pytest.skip("OpenTelemetry not installed")

    recorder.clear()
    with mock.patch.object(
        tel_module._TRACER, "start_as_current_span"
    ) as mock_start:
        mock_span = mock.MagicMock()
        mock_start.return_value.__enter__.return_value = mock_span

        with traced_span("test_op", attributes={"key": "value"}):
            pass

        mock_start.assert_called_once_with("test_op")
        mock_span.set_attribute.assert_called_with("key", "value")


def test_traced_span_otel_fallback_when_unavailable():
    """Verify fallback behavior when OTel is not available."""
    import omniai.telemetry as tel_module

    original_tracer = tel_module._TRACER
    tel_module._TRACER = None
    try:
        recorder.clear()
        with traced_span("fallback_op", attributes={"mode": "fallback"}):
            pass

        assert len(recorder.spans) == 1
        assert recorder.spans[0].attributes["mode"] == "fallback"
        # No OTel span was created, but recording still works
    finally:
        tel_module._TRACER = original_tracer


def test_traced_span_otel_attribute_error_suppression():
    """Verify OTel attribute setting errors don't break the span."""
    import omniai.telemetry as tel_module

    if tel_module._TRACER is None:
        pytest.skip("OpenTelemetry not installed")

    recorder.clear()
    with mock.patch.object(
        tel_module._TRACER, "start_as_current_span"
    ) as mock_start:
        mock_span = mock.MagicMock()
        mock_span.set_attribute.side_effect = RuntimeError("OTel error")
        mock_start.return_value.__enter__.return_value = mock_span

        # Should not raise even though set_attribute failed
        with traced_span("op", attributes={"key": "value"}):
            pass

        assert len(recorder.spans) == 1
        assert recorder.spans[0].name == "op"


# -- Recorder isolation tests -----------------------------------------------


def test_recorder_clear_isolation():
    """Verify recorder.clear() doesn't affect concurrent operations."""
    recorder.clear()
    with traced_span("op1"):
        pass
    assert len(recorder.spans) == 1

    recorder.clear()
    assert len(recorder.spans) == 0

    with traced_span("op2"):
        pass
    assert len(recorder.spans) == 1
    assert recorder.spans[0].name == "op2"


def test_multiple_recorders_independent():
    """Verify multiple TelemetryRecorder instances are independent."""
    rec1 = TelemetryRecorder()
    rec2 = TelemetryRecorder()

    rec1.record(SpanRecord(name="rec1_op"))
    rec2.record(SpanRecord(name="rec2_op"))

    assert len(rec1.spans) == 1
    assert len(rec2.spans) == 1
    assert rec1.spans[0].name == "rec1_op"
    assert rec2.spans[0].name == "rec2_op"


# -- Edge cases and special characters ------------------------------------------


def test_traced_span_empty_name():
    """Verify empty span names are handled."""
    recorder.clear()
    with traced_span(""):
        pass
    assert recorder.spans[0].name == ""


def test_traced_span_special_characters_in_name():
    """Verify special characters in span names are preserved."""
    recorder.clear()
    special_names = [
        "op:with:colons",
        "op-with-dashes",
        "op/with/slashes",
        "op.with.dots",
        "op(with)parens",
    ]
    for name in special_names:
        with traced_span(name):
            pass

    assert [s.name for s in recorder.spans] == special_names


def test_traced_span_special_characters_in_attributes():
    """Verify special characters in attributes are preserved."""
    recorder.clear()
    with traced_span("op", attributes={"error": "msg: failed!", "path": "/a/b/c"}):
        pass

    attrs = recorder.spans[0].attributes
    assert attrs["error"] == "msg: failed!"
    assert attrs["path"] == "/a/b/c"


def test_traced_span_large_attributes():
    """Verify large attribute values are handled."""
    recorder.clear()
    large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
    with traced_span("op", attributes=large_dict):
        pass

    assert len(recorder.spans[0].attributes) == 100


def test_traced_span_various_attribute_types():
    """Verify various Python types can be stored as attributes."""
    recorder.clear()
    with traced_span("op", attributes={
        "int": 42,
        "float": 3.14,
        "str": "text",
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "tuple": (1, 2),
    }):
        pass

    attrs = recorder.spans[0].attributes
    assert attrs["int"] == 42
    assert attrs["float"] == 3.14
    assert attrs["str"] == "text"
    assert attrs["bool"] is True
    assert attrs["none"] is None
    assert attrs["list"] == [1, 2, 3]
    assert attrs["dict"] == {"nested": "value"}
    assert attrs["tuple"] == (1, 2)
