import time
import threading
from typing import Dict, Tuple
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.config import settings

class TokenBucket:
    def __init__(self, rate_per_min: int, burst: int):
        self.rate = rate_per_min / 60.0
        self.capacity = float(burst)
        self.tokens = self.capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def allow(self) -> Tuple[bool, float]:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            # сколько ждать до 1 токена
            deficit = 1.0 - self.tokens
            wait = deficit / self.rate if self.rate > 0 else 60.0
            return False, wait

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.buckets: Dict[str, TokenBucket] = {}

    async def dispatch(self, request: Request, call_next):
        # можно ключом сделать API-key или IP(simply)
        if request.url.path in ("/healthz", "/metrics"):
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        b = self.buckets.get(ip)
        if b is None:
            b = self.buckets.setdefault(ip, TokenBucket(settings.rate_limit_per_min, settings.rate_limit_burst))
        allowed, wait = b.allow()
        if not allowed:
            resp = JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests", "retry_after": round(wait, 2)}
            )
            resp.headers["Retry-After"] = str(int(wait + 0.5))
            return resp
        return await call_next(request)
