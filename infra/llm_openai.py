from __future__ import annotations
from typing import List
from domain.llm import ILLMProvider, Msg, LLMResult
from app.config import settings

class OpenAILLM(ILLMProvider):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # type: ignore
        # ВАЖНО: короткий таймаут, чтобы не «висло»
        self.client = OpenAI(
            api_key=settings.openai_api_key or "EMPTY",
            base_url=(settings.openai_base_url or "").rstrip("/") or None,
            timeout=30.0,  # ← вот это добавили
        )
        self.model = model or settings.llm_model

    def generate(self, messages: List[Msg], temperature: float = 0.3) -> LLMResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            stream=False,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": getattr(choice, "finish_reason", "") or "",
        }