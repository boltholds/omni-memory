from __future__ import annotations

from dataclasses import dataclass

from omni_memory.config import settings
from omni_memory.domain.llm import ILLMProvider


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = "none"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0


def build_llm(config: LLMConfig | None = None) -> ILLMProvider | None:
    """Build an LLM provider from explicit config or environment settings.

    This factory is the BYO-LLM boundary. Application code can skip it entirely
    and pass its own ILLMProvider into OmniMemory/ModelBundle.
    """
    cfg = config or LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    provider = (cfg.provider or "none").lower().replace("-", "_")

    if provider in {"", "none", "off", "disabled"}:
        return None

    if provider in {"openai", "openai_compatible", "compatible"}:
        if not cfg.model:
            raise RuntimeError("LLM model is required")
        from omni_memory.infra.llm.llm_openai_compatible import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            model=cfg.model,
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )

    if provider == "ollama":
        from omni_memory.infra.llm.llm_ollama import OllamaLLM

        return OllamaLLM(
            model=cfg.model or settings.llm_ollama_model,
            base_url=cfg.base_url or settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
