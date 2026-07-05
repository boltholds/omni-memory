from __future__ import annotations

import logging
import time
from typing import List

import httpx

from omni_memory.config import settings
from omni_memory.domain.llm import ILLMProvider, Msg, LLMResult

log = logging.getLogger("app.llm")


class OllamaLLM(ILLMProvider):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_ollama_model
        self.base = _native_ollama_base_url(base_url or settings.ollama_base_url)

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
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=2.0)) as client:
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


def _native_ollama_base_url(base_url: str | None) -> str:
    base = (base_url or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3].rstrip("/")
    return base
