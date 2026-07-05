from __future__ import annotations

from omni_memory.config import settings
from omni_memory.infra.llm.llm_openai_compatible import OpenAICompatibleLLM


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI adapter kept for backwards compatibility."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            model=model or settings.llm_model,
            api_key=api_key if api_key is not None else settings.openai_api_key,
            base_url=base_url if base_url is not None else settings.openai_base_url,
            timeout=timeout,
        )
