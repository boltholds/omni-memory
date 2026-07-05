from __future__ import annotations
import json
import logging
import logging.handlers
import sys
import re
from datetime import datetime, UTC
from typing import Any, Dict
from omni_memory.config import settings

EmailRe = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ApiKeyLikeRe = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}", re.I)

def _redact(s: str) -> str:
    s = EmailRe.sub("[REDACTED_EMAIL]", s)
    s = ApiKeyLikeRe.sub("[REDACTED_SECRET]", s)
    return s

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # добавим стандартные поля, если есть
        for f in ("request_id", "path", "method", "status", "duration_ms", "client", "user_agent"):
            v = getattr(record, f, None)
            if v is not None:
                payload[f] = v
        for key, value in _structured_extra(record).items():
            if key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        if settings.trace_redact:
            line = _redact(line)
        return line


_LOG_RECORD_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def _structured_extra(record: logging.LogRecord) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _LOG_RECORD_RESERVED and not key.startswith("_")
    }

def setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    if settings.log_json:
        fmt = JsonFormatter()
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if settings.log_file:
        fh = logging.handlers.RotatingFileHandler(
            settings.log_file, maxBytes=settings.log_rotation_mb * 1024 * 1024, backupCount=settings.log_keep_files, encoding="utf-8"
        )
        if settings.log_json:
            fh.setFormatter(JsonFormatter())
        else:
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
        root.addHandler(fh)

    # Утихомирим шумные логгеры
    for noisy in ("uvicorn.access", "uvicorn.error", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(level)
