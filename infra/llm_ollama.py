# infra/llm_ollama.py
from __future__ import annotations
from typing import List
import httpx
from domain.llm import ILLMProvider, Msg, LLMResult
from app.config import settings
import logging, time
log = logging.getLogger("app.llm")

class OllamaLLM(ILLMProvider):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_ollama_model
        self.base = (base_url or settings.ollama_base_url).rstrip("/")

    def generate(self, messages: List[Msg], temperature: float = 0.3) -> LLMResult:
        url = f"{self.base}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature), "stream": False}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json=payload, headers=headers)
                dur = int((time.perf_counter() - t0) * 1000)
                r.raise_for_status()
                data = r.json()
                ch = data["choices"][0]
                text = ch.get("message", {}).get("content", "") or ch.get("text", "") or ""
                usage = data.get("usage") or {}
                log.info("llm_call_ok", extra={
                    "model": self.model,
                    "duration_ms": dur,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                })
                return {
                    "text": text,
                    "model": self.model,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish_reason": ch.get("finish_reason", ""),
                }
        except Exception as e:
            dur = int((time.perf_counter() - t0) * 1000)
            log.exception("llm_call_failed", extra={"model": self.model, "duration_ms": dur})
            raise
