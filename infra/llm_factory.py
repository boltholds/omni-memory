# infra/llm_factory.py
from __future__ import annotations
from domain.llm import ILLMProvider
from app.config import settings

def build_llm() -> ILLMProvider | None:
    prov = (settings.llm_provider or "none").lower()
    if prov == "openai":
        # ключ обязателен
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY не задан")
        from infra.llm_openai import OpenAILLM
        return OpenAILLM(model=settings.llm_model)
    if prov == "ollama":
        from infra.llm_ollama import OllamaLLM
        return OllamaLLM(model=settings.llm_ollama_model, base_url=settings.ollama_base_url)
    # none
    return None
