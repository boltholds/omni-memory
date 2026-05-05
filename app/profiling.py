from __future__ import annotations
import time
import logging
import functools
from typing import Callable, TypeVar, Any, Optional

T = TypeVar("T")
log = logging.getLogger("app.prof")

def timed(name: Optional[str] = None, slow_ms: int = 200) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Замеряет длительность функции. Пишет log.info при ok, log.warning если > slow_ms.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        tag = name or f"{fn.__module__}.{fn.__name__}"
        @functools.wraps(fn)
        def wrap(*args: Any, **kwargs: Any) -> T:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dur = int((time.perf_counter() - t0) * 1000)
                if dur >= slow_ms:
                    log.warning("slow", extra={"op": tag, "duration_ms": dur})
                else:
                    log.info("ok", extra={"op": tag, "duration_ms": dur})
        return wrap
    return deco
