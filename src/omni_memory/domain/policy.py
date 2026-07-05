# domain/policy.py
from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, Field
import re
import time


class ConfidencePolicy(BaseModel):
    accept: float = 0.6
    reject: float = 0.3


class TTLPolicy(BaseModel):
    high_volatility_days: int = 7
    normal_days: int = 365


class MemoryPolicy(BaseModel):
    confidence: ConfidencePolicy = Field(default_factory=ConfidencePolicy)
    ttl: TTLPolicy = Field(default_factory=TTLPolicy)

    # --- PII фильтры ---
    _email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    _apikey_re = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}", re.I)

    def filter_write(self, obj: Dict[str, Any]) -> bool:
        """
        Возвращает True, если можно записывать объект.
        - Отбрасываем текстовые заметки с PII (email, ключи).
        """
        text = ""
        if isinstance(obj, dict):
            if "payload" in obj and isinstance(obj["payload"], dict):
                text = str(obj["payload"].get("text", ""))
            elif "text" in obj:
                text = str(obj["text"])
            elif "content" in obj:
                text = str(obj["content"])
        if self._email_re.search(text) or self._apikey_re.search(text):
            return False
        return True

    def apply_ttl(self, obj: Dict[str, Any]) -> float:
        """
        Возвращает timestamp истечения TTL.
        - Если meta.volatility == "high" → 7 дней.
        - Иначе → 365 дней.
        """
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        volatility = str(meta.get("volatility", "normal")).lower()
        days = (
            self.ttl.high_volatility_days
            if volatility == "high"
            else self.ttl.normal_days
        )
        return time.time() + days * 86400.0
