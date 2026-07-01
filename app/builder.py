from __future__ import annotations

from domain.llm import ILLMProvider
from domain.model_ports import IEmbedder, ModelBundle
from domain.repositories import IFactRepo, IVectorRepo
from app.memory import OmniMemory

from infra.repo.episodic_repo import EpisodicRepo


def build_memory(
    *,
    use_llm: bool = False,
    reject_conflicts: bool = False,
    llm: ILLMProvider | None = None,
    embedder: IEmbedder | None = None,
    model_bundle: ModelBundle | None = None,
    vector_repo: IVectorRepo | None = None,
    graph_repo: IFactRepo | None = None,
    episodic_repo: EpisodicRepo | None = None,
) -> OmniMemory:
    """Build the central OmniMemory facade used by CLI, FastAPI and examples.

    BYO-LLM:
        build_memory(llm=my_llm)

    BYO-Embedder:
        build_memory(embedder=my_embedder)

    Full BYOM:
        build_memory(model_bundle=ModelBundle(...))

    Advanced/tests/CLI:
        build_memory(vector_repo=..., graph_repo=..., episodic_repo=...)
    """
    return OmniMemory(
        use_llm=use_llm,
        reject_conflicts=reject_conflicts,
        llm=llm,
        embedder=embedder,
        model_bundle=model_bundle,
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        episodic_repo=episodic_repo,
    )
    
    

