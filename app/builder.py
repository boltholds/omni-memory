from __future__ import annotations

from domain.llm import ILLMProvider
from domain.model_ports import IEmbedder, ModelBundle
from app.memory import OmniMemory


def build_memory(
    *,
    use_llm: bool = False,
    reject_conflicts: bool = False,
    llm: ILLMProvider | None = None,
    embedder: IEmbedder | None = None,
    model_bundle: ModelBundle | None = None,
) -> OmniMemory:
    """Build the central OmniMemory facade used by CLI, FastAPI and examples.

    For BYO-LLM pass llm=... .
    For BYOM pass model_bundle=ModelBundle(llm=..., embedder=..., reranker=..., distiller=...).
    """
    return OmniMemory(
        use_llm=use_llm,
        reject_conflicts=reject_conflicts,
        llm=llm,
        embedder=embedder,
        model_bundle=model_bundle,
    )
