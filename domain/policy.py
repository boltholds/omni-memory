from __future__ import annotations
from typing import Dict
from pydantic import BaseModel

class ConfidenceThresholds(BaseModel):
    accept: float = 0.6
    warn: float = 0.4

class MemoryPolicy(BaseModel):
    max_context_tokens: int = 2048
    confidence: ConfidenceThresholds = ConfidenceThresholds()
    ttl_days_by_volatility: Dict[str, int] = {"high": 7, "normal": 90, "low": 365}
    pii_blocklist: list[str] = ["@","api_key","password"]
