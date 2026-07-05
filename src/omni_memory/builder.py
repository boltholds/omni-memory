from __future__ import annotations

from omni_memory.domain.llm import ILLMProvider
from omni_memory.domain.model_ports import IEmbedder, ModelBundle
from omni_memory.domain.repositories import IFactRepo, IVectorRepo
from omni_memory.domain.distiller import ISessionMemoryDistiller
from omni_memory.domain.experience_evaluator import ExperienceEvaluator
from omni_memory.fact_mining import FactExtractor
from omni_memory.memory import OmniMemory

from omni_memory.infra.graph_backend import GraphBackend
from omni_memory.infra.record_store import RecordStoreBackends
from omni_memory.infra.repo.episodic_repo import EpisodicRepo
from omni_memory.infra.repo.decision_repo import DecisionRepo
from omni_memory.infra.repo.experience_repo import ExperienceRepo
from omni_memory.infra.repo.cognitive_repo import FailurePatternRepo, SkillRepo
from omni_memory.infra.repo.review_repo import ReviewQueueRepo
from omni_memory.infra.rerankers import build_reranker
from omni_memory.infra.vector_index import VectorIndexBackend


def build_memory(
    *,
    use_llm: bool = False,
    reject_conflicts: bool = False,
    llm: ILLMProvider | None = None,
    embedder: IEmbedder | None = None,
    model_bundle: ModelBundle | None = None,
    distiller: ISessionMemoryDistiller | None = None,
    vector_repo: IVectorRepo | None = None,
    vector_index_backend: VectorIndexBackend | None = None,
    graph_repo: IFactRepo | None = None,
    graph_backend: GraphBackend | None = None,
    record_store_backends: RecordStoreBackends | None = None,
    episodic_repo: EpisodicRepo | None = None,
    decision_repo: DecisionRepo | None = None,
    experience_repo: ExperienceRepo | None = None,
    skill_repo: SkillRepo | None = None,
    failure_pattern_repo: FailurePatternRepo | None = None,
    review_queue_repo: ReviewQueueRepo | None = None,
    fact_extractor: FactExtractor | None = None,
    experience_evaluator: ExperienceEvaluator | None = None,
) -> OmniMemory:
    """Build the central OmniMemory facade used by CLI, FastAPI and examples.

    BYO-LLM:
        build_memory(llm=my_llm)

    BYO-Embedder:
        build_memory(embedder=my_embedder)

    BYO vector index backend:
        build_memory(vector_index_backend=my_vector_index)

    BYO graph backend:
        build_memory(graph_backend=my_graph_backend)

    BYO typed record stores:
        build_memory(record_store_backends=RecordStoreBackends(...))

    Full BYOM:
        build_memory(model_bundle=ModelBundle(...))

    Domain evaluators:
        build_memory(experience_evaluator=my_domain_router)

    Advanced/tests/CLI:
        build_memory(vector_repo=..., graph_repo=..., episodic_repo=...)
    """
    selected_bundle = model_bundle
    if selected_bundle is None:
        selected_bundle = ModelBundle(reranker=build_reranker())
    elif selected_bundle.reranker is None:
        selected_bundle = ModelBundle(
            llm=selected_bundle.llm,
            embedder=selected_bundle.embedder,
            reranker=build_reranker(),
            distiller=selected_bundle.distiller,
        )

    record_backends = record_store_backends or RecordStoreBackends()

    return OmniMemory(
        use_llm=use_llm,
        reject_conflicts=reject_conflicts,
        llm=llm,
        embedder=embedder,
        model_bundle=selected_bundle,
        distiller=distiller,
        vector_repo=vector_repo,
        vector_index_backend=vector_index_backend,
        graph_repo=graph_repo,
        graph_backend=graph_backend,
        episodic_repo=episodic_repo,
        decision_repo=decision_repo or (DecisionRepo(backend=record_backends.decision) if record_backends.decision is not None else None),
        experience_repo=experience_repo or (ExperienceRepo(backend=record_backends.experience) if record_backends.experience is not None else None),
        skill_repo=skill_repo or (SkillRepo(backend=record_backends.skill) if record_backends.skill is not None else None),
        failure_pattern_repo=failure_pattern_repo or (FailurePatternRepo(backend=record_backends.failure_pattern) if record_backends.failure_pattern is not None else None),
        review_queue_repo=review_queue_repo or (ReviewQueueRepo(backend=record_backends.review_queue) if record_backends.review_queue is not None else None),
        fact_extractor=fact_extractor,
        experience_evaluator=experience_evaluator,
    )
