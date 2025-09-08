# infra/llm_ollama.py
from __future__ import annotations
from typing import List
import httpx
from domain.llm import ILLMProvider, Msg, LLMResult
from app.config import settings

class OllamaLLM(ILLMProvider):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_ollama_model
        self.base = (base_url or settings.ollama_base_url).rstrip("/")

    def generate(self, messages: List[Msg], temperature: float = 0.3) -> LLMResult:
        # склеим историю в один prompt (просто и совместимо)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        payload = {"model": self.model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{self.base}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        return {
            "text": data.get("response", "") or "",
            "model": self.model,
            "finish_reason": "stop",
        }
