# infra/metrics.py
from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any

class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "requests_total": 0,
            "retrieve_calls": 0,
            "context_calls": 0,
            "writeback_saved": 0,
            "writeback_rejected": 0,
            "conflicts_detected": 0,
        }
        # простая EMA латентности всех запросов
        self._lat_ema_ms: float = 0.0
        self._ema_alpha: float = 0.2

    def inc(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def add_latency(self, ms: float) -> None:
        with self._lock:
            if self._lat_ema_ms == 0.0:
                self._lat_ema_ms = ms
            else:
                a = self._ema_alpha
                self._lat_ema_ms = a * ms + (1 - a) * self._lat_ema_ms

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._counters,
                "latency_ema_ms": round(self._lat_ema_ms, 2),
            }

metrics = _Registry()

@contextmanager
def timeit_request():
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        metrics.add_latency(dt_ms)
