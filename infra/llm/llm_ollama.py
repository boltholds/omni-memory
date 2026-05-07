from __future__ import annotations

import logging
import time
from typing import List

import httpx

from app.config import settings
from domain.llm import ILLMProvider, Msg, LLMResult

log = logging.getLogger("app.llm")


class OllamaLLM(ILLMProvider):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_ollama_model
        self.base = (base_url or settings.ollama_base_url or "http://localhost:11434/v1").rstrip("/")

    def generate(self, messages: List[Msg], temperature: float | None = None) -> LLMResult:
        url = f"{self.base}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": 0.3 if temperature is None else float(temperature)},
            "stream": False,
        }

        t0 = time.perf_counter()

        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(url, json=payload)
                dur = int((time.perf_counter() - t0) * 1000)
                r.raise_for_status()
                data = r.json()

            text = data.get("message", {}).get("content", "") or ""

            log.info("llm_call_ok", extra={
                "model": self.model,
                "duration_ms": dur,
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            })

            return {
                "text": text,
                "model": self.model,
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "finish_reason": data.get("done_reason", ""),
            }

        except Exception:
            dur = int((time.perf_counter() - t0) * 1000)
            log.exception("llm_call_failed", extra={"model": self.model, "duration_ms": dur})
            raise
