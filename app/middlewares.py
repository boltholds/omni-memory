from __future__ import annotations
import logging, time, uuid
from typing import Callable
from fastapi import Request, Response
from app.config import settings
from app.logging import _redact

log = logging.getLogger("app.http")

async def tracing_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    rid = request.headers.get(settings.header_request_id) or str(uuid.uuid4())
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
    response.headers[settings.header_request_id] = rid

    extra = {**base, "status": response.status_code, "duration_ms": duration_ms}
    log.info("request", extra=extra)

    # тело ответа опционально
    if settings.trace_log_body and sampled:
        try:
            # response.body может быть не доступно — не тратим время, просто логируем статус
            pass
        except Exception:
            pass

    return response
