from __future__ import annotations

from dataclasses import dataclass

from omni_memory.config import settings
from omni_memory.domain.distiller import ISessionMemoryDistiller
from omni_memory.domain.llm import ILLMProvider
from omni_memory.infra.distillers.session_llm import ConservativeLLMSessionDistiller
from omni_memory.infra.llm.llm_factory import LLMConfig, build_llm


@dataclass(frozen=True, slots=True)
class SessionDistillerConfig:
    provider: str = "inherit"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0


def build_session_distiller(
    config: SessionDistillerConfig | None = None,
    *,
    existing_llm: ILLMProvider | None = None,
) -> ISessionMemoryDistiller | None:
    cfg = config or SessionDistillerConfig(
        provider=settings.distiller_provider,
        model=settings.distiller_model,
        api_key=settings.distiller_api_key,
        base_url=settings.distiller_base_url,
        temperature=settings.distiller_temperature,
    )

    provider = (cfg.provider or "inherit").lower().replace("-", "_")
    if provider in {"", "none", "off", "disabled"}:
        return None

    if provider == "inherit":
        if existing_llm is None:
            return None
        return ConservativeLLMSessionDistiller(existing_llm, temperature=cfg.temperature)

    llm = build_llm(
        LLMConfig(
            provider=provider,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
    )
    if llm is None:
        return None
    return ConservativeLLMSessionDistiller(llm, temperature=cfg.temperature)
