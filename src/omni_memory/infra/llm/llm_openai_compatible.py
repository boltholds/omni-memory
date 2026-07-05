from __future__ import annotations

import logging
import time
from typing import List

from openai import OpenAI  # type: ignore

from omni_memory.metrics import LLM_CALLS, LLM_LATENCY
from omni_memory.stats import stats
from omni_memory.domain.llm import ILLMProvider, LLMResult, Msg

log = logging.getLogger("app.llm")


class OpenAICompatibleLLM(ILLMProvider):
    """LLM adapter for OpenAI-compatible Chat Completions APIs.

    Works with OpenAI, vLLM, LM Studio, Ollama OpenAI-compatible endpoint,
    llama.cpp server, LocalAI and internal gateways that implement /v1/chat/completions.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "EMPTY",
        base_url: str | None = None,
        timeout: float = 240.0,
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=(base_url or "").rstrip("/") or None,
            timeout=timeout,
        )
        self.status = "init"

    def generate(self, messages: List[Msg], temperature: float | None = None) -> LLMResult:
        t0 = time.perf_counter()
        stop_llm = stats.timeit("llm.call_ms")
        self.status = "ok"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.3 if temperature is None else float(temperature),
                stream=False,
            )
            dur = int((time.perf_counter() - t0) * 1000)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = getattr(resp, "usage", None)

            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            finish_reason = str(getattr(choice, "finish_reason", "") or "")

            log.info(
                "llm_call_ok",
                extra={
                    "model": self.model,
                    "duration_ms": dur,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )

            return {
                "text": text,
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
            }
        except Exception:
            self.status = "error"
            dur = int((time.perf_counter() - t0) * 1000)
            log.exception("llm_call_failed", extra={"model": self.model, "duration_ms": dur})
            raise
        finally:
            dur = int((time.perf_counter() - t0) * 1000)
            stop_llm()
            LLM_CALLS.labels(self.model, self.status).inc()
            LLM_LATENCY.labels(self.model, self.status).observe(dur)
