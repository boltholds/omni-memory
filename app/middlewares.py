from __future__ import annotations
import logging, time, uuid
from typing import Callable
from fastapi import Request, Response

import uuid, time, logging
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

from app.config import settings
from app.logging import _redact
from app.metrics import HTTP_REQUESTS, HTTP_LATENCY

log = logging.getLogger("app.http")


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Нормализованный шаблон пути ("/answer", "/context", "/admin/reset", ...)
        route_pattern = request.url.path
        if route_pattern == "/metrics":
            return await call_next(request)
        for route in request.app.router.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL and hasattr(route, "path_format"):
                route_pattern = route.path_format
                break

        method = request.method
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            dur = time.perf_counter() - t0
            # Записываем длительность уже с реальным статусом
            HTTP_LATENCY.labels(method, route_pattern, status).observe(dur)
            HTTP_REQUESTS.labels(method, route_pattern, status).inc()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(settings.request_id_header) or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            extra = {
                "request_id": rid, "path": str(request.url.path),
                "method": request.method, "status": getattr(response, "status_code", 0),
                "duration_ms": duration_ms, "client": request.client.host if request.client else "",
                "user_agent": request.headers.get("user-agent",""),
            }
            log.info("request", extra=extra)
        response.headers[settings.request_id_header] = rid
        return response


async def tracing_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    rid = request.headers.get(settings.request_id_header) or str(uuid.uuid4())
    # положим в state
    request.state.request_id = rid

    # базовые поля
    base = {
        "request_id": rid,
        "path": request.url.path,
        "method": request.method,
        "client": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent", ""),
    }

    # опционально тело запроса (сэмплинг)
    sampled = (settings.trace_sample_rate >= 1.0) or (uuid.uuid4().int % int(1/settings.trace_sample_rate or 1) == 0)
    if settings.trace_log_body and sampled:
        try:
            body = (await request.body()).decode("utf-8", errors="ignore")
            if settings.trace_redact:
                body = _redact(body)
            if len(body) > settings.trace_body_max:
                body = body[:settings.trace_body_max] + "…"
            log.debug("request body", extra={**base, "msg": body})
        except Exception:
            pass

    # пропишем rid в ответ
    try:
        response: Response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start)*1000)
        log.exception("unhandled exception", extra={**base, "duration_ms": duration_ms, "status": 500})
        raise

    duration_ms = int((time.perf_counter() - start)*1000)
    response.headers[settings.request_id_header] = rid

    extra = {**base, "status": response.status_code, "duration_ms": duration_ms}
    log.info("request", extra=extra)

    # тело ответа опционально
    if settings.trace_log_body and sampled:
        try:
            # response.body может быть не доступно — не тратим время, просто логируем статус
            pass
        except Exception:
            pass

    SLA_MS = {
    "/answer": 6000, # было 1200 — поднимаем до 6s под текущую модель
    "/retrieve": 300,
    "/context": 500,
    }
    thr = SLA_MS.get(request.url.path)
    if thr and duration_ms > thr:
        log.warning("slow_endpoint", extra={**extra, "sla_ms": thr})
    
    return response
