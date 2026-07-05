from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from omni_memory.config import settings


log = logging.getLogger("app.telemetry")

_tracer: Any | None = None
_setup_attempted = False


def setup_telemetry() -> bool:
    """Configure OpenTelemetry when explicitly enabled.

    The default path is deliberately dependency-light: when telemetry is disabled
    or OpenTelemetry packages are unavailable, this module behaves as a no-op.
    """

    global _tracer, _setup_attempted
    if _tracer is not None:
        return True
    if _setup_attempted:
        return False
    _setup_attempted = True

    if not settings.omni_telemetry_enabled:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception as exc:
        log.warning(
            "telemetry_setup_unavailable",
            exc_info=True,
            extra={
                "component": "telemetry",
                "op": "setup_telemetry",
                "error_type": type(exc).__name__,
                "fallback": "noop",
            },
        )
        return False

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.omni_otel_service_name,
                "deployment.environment": settings.env,
            }
        )
    )
    exporter_name = settings.omni_otel_exporter.strip().casefold()
    try:
        if exporter_name == "otlp":
            exporter = _build_otlp_exporter(settings.omni_otel_endpoint)
        elif exporter_name == "console":
            exporter = ConsoleSpanExporter()
        else:
            raise ValueError(f"Unsupported telemetry exporter: {settings.omni_otel_exporter}")
    except Exception as exc:
        log.warning(
            "telemetry_exporter_unavailable",
            exc_info=True,
            extra={
                "component": "telemetry",
                "op": "setup_telemetry",
                "exporter": settings.omni_otel_exporter,
                "endpoint": settings.omni_otel_endpoint,
                "error_type": type(exc).__name__,
                "fallback": "noop",
            },
        )
        return False

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("omni-memory")
    log.info(
        "telemetry_enabled",
        extra={
            "component": "telemetry",
            "op": "setup_telemetry",
            "exporter": exporter_name,
            "endpoint": settings.omni_otel_endpoint,
            "service_name": settings.omni_otel_service_name,
        },
    )
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any | None]:
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as active_span:
        for key, value in attributes.items():
            _set_attribute(active_span, key, value)
        try:
            yield active_span
        except Exception as exc:
            _record_exception(active_span, exc)
            raise


def _get_tracer() -> Any | None:
    if _tracer is not None:
        return _tracer
    if not settings.omni_telemetry_enabled:
        return None
    return _tracer if setup_telemetry() else None


def _build_otlp_exporter(endpoint: str | None) -> Any:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except Exception:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if endpoint:
        return OTLPSpanExporter(endpoint=endpoint)
    return OTLPSpanExporter()


def _set_attribute(active_span: Any, key: str, value: Any) -> None:
    if value is None or not hasattr(active_span, "set_attribute"):
        return
    active_span.set_attribute(key, _attribute_value(value))


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
        return list(value)
    return str(value)


def _record_exception(active_span: Any, exc: Exception) -> None:
    if hasattr(active_span, "record_exception"):
        active_span.record_exception(exc)
    try:
        from opentelemetry.trace import Status, StatusCode

        if hasattr(active_span, "set_status"):
            active_span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        return
