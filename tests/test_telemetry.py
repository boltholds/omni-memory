from __future__ import annotations

import pytest

from omni_memory import telemetry


class FakeSpan:
    def __init__(self) -> None:
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def set_status(self, status) -> None:
        self.status = status


class FakeSpanContext:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeTracer:
    def __init__(self) -> None:
        self.started = []

    def start_as_current_span(self, name: str):
        span = FakeSpan()
        self.started.append((name, span))
        return FakeSpanContext(span)


def test_span_is_noop_when_telemetry_is_disabled(monkeypatch):
    monkeypatch.setattr(telemetry.settings, "omni_telemetry_enabled", False)
    monkeypatch.setattr(telemetry, "_tracer", None)

    with telemetry.span("memory.retrieve", intent="write_code") as span:
        assert span is None


def test_span_records_structured_attributes_with_fake_tracer(monkeypatch):
    fake = FakeTracer()
    monkeypatch.setattr(telemetry, "_tracer", fake)

    with telemetry.span("memory.retrieve", intent="write_code", item_count=3, payload={"a": 1}):
        pass

    assert fake.started[0][0] == "memory.retrieve"
    span = fake.started[0][1]
    assert span.attributes["intent"] == "write_code"
    assert span.attributes["item_count"] == 3
    assert span.attributes["payload"] == "{'a': 1}"


def test_span_records_exception_before_reraising(monkeypatch):
    fake = FakeTracer()
    monkeypatch.setattr(telemetry, "_tracer", fake)

    with pytest.raises(RuntimeError):
        with telemetry.span("memory.write"):
            raise RuntimeError("boom")

    span = fake.started[0][1]
    assert len(span.exceptions) == 1
    assert str(span.exceptions[0]) == "boom"
