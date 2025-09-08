from __future__ import annotations
from typing import List
from domain.llm import ILLMProvider, Msg, LLMResult
from app.config import settings
from app.stats import stats
import logging, time
log = logging.getLogger("app.llm")


class OpenAILLM(ILLMProvider):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(
            api_key=settings.openai_api_key or "EMPTY",
            base_url=(settings.openai_base_url or "").rstrip("/") or None,
            timeout=30.0,  # короткий таймаут
        )
        self.model = model or settings.llm_model

    def generate(self, messages: List[Msg], temperature: float = 0.3) -> LLMResult:
        t0 = time.perf_counter()
        stop_llm = stats.timeit("llm.call_ms")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=temperature,
                stream=False,
            )
            dur = int((time.perf_counter() - t0) * 1000)

            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = getattr(resp, "usage", None) or {}

            log.info("llm_call_ok", extra={
                "model": self.model,
                "duration_ms": dur,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            })

            return {
                "text": text,
                "model": self.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "finish_reason": getattr(choice, "finish_reason", "") or "",
            }
        except Exception as e:
            dur = int((time.perf_counter() - t0) * 1000)
            log.exception("llm_call_failed", extra={
                "model": self.model,
                "duration_ms": dur,
            })
            raise
        finally:
            stop_llm()