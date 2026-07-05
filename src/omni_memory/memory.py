from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_memory.config import settings
from omni_memory.infra.embeddings import build_embedder
from omni_memory.orchestrator import Orchestrator
from omni_memory.prompting import PromptRenderer
from omni_memory.retriever import Retriever
from omni_memory.agent_cycle import AgentCycleRecord
from omni_memory.consolidation import ConsolidationResult, ExperienceConsolidator
from omni_memory.development_cycle import DevelopmentCycleDraft, DevelopmentCycleRecorder
from omni_memory.development_memory_workflow import DevelopmentMemoryWorkflow, FinishDevelopmentTaskResult
from omni_memory.fact_maintenance import FactMaintenanceCommand, FactMaintenanceResult, FactMaintenanceService
from omni_memory.fact_mining import FactExtractor, FactMiningResult, FactMiningService
from omni_memory.memory_commands import (
    RecordAgentCycleCommand,
    RecordExperienceCommand,
    WriteFailurePatternCommand,
    WriteDecisionCommand,
    WriteFactCommand,
    WriteItemsCommand,
    WriteNoteCommand,
    WriteSkillCommand,
)
from omni_memory.memory_repositories import MemoryClearCommand, MemoryClearReport, MemoryRepositories, build_memory_repositories
from omni_memory.ops_cycle import OpsCycleDraft
from omni_memory.ops_memory_workflow import OpsMemoryWorkflow
from omni_memory.review_queue import ReviewActionResult, ReviewQueueService
from omni_memory.domain.distiller import ISessionMemoryDistiller, SessionTurn
from omni_memory.domain.experience_evaluator import ExperienceEvaluator
from omni_memory.domain.models import ContextPack, ConflictReport, DecisionRecord, ExperienceRecord, RetrievalBundle, ReviewItem, WriteReport
from omni_memory.domain.model_ports import IEmbedder, ModelBundle
from omni_memory.domain.policy import MemoryPolicy
from omni_memory.domain.repositories import IFactRepo, IVectorRepo
from omni_memory.domain.writeback import WritebackRequest, WritebackResult
from omni_memory.session_distillation import ConservativeCandidateValidator, accepted_candidates, build_transcript, candidates_to_writeback_items
from omni_memory.telemetry import span as telemetry_span
from omni_memory.infra.consistency import SimpleConsistencyEngine
from omni_memory.infra.distillers.factory import build_session_distiller
from omni_memory.infra.llm.llm_factory import build_llm
from omni_memory.infra.repo.decision_repo import DecisionRepo
from omni_memory.infra.repo.episodic_repo import EpisodicRepo
from omni_memory.infra.repo.experience_repo import ExperienceRepo
from omni_memory.infra.repo.cognitive_repo import FailurePatternRepo, SkillRepo
from omni_memory.infra.repo.review_repo import ReviewQueueRepo
from omni_memory.infra.vector_index import VectorIndexBackend
from omni_memory.writeback.memory_policies import ConfidenceConfig, ConfidencePolicy, ConflictPolicy, DedupPolicy, PiiPolicy, ProvenancePolicy, TTLConfig, TTLPolicy
from omni_memory.writeback.service import MemoryRepositoryRouter, WriteBackService
from omni_memory.writeback.writeback_policies import (
    DecisionWritebackPolicy,
    EpisodeWritebackPolicy,
    ExperienceWritebackPolicy,
    FactWritebackPolicy,
    FailurePatternWritebackPolicy,
    NoteWritebackPolicy,
    PreferenceWritebackPolicy,
    SkillWritebackPolicy,
    WritebackPolicyResolver,
)


@dataclass(frozen=True)
class MemoryAnswer:
    answer: str
    advisories: list[str]
    used_sections: list[str]
    context: dict[str, Any]
    model: str | None = None


def build_writeback_service(*, repositories: MemoryRepositories, reject_conflicts: bool = False) -> WriteBackService:
    memory_policy = MemoryPolicy()
    resolver = WritebackPolicyResolver(
        [
            FactWritebackPolicy(),
            EpisodeWritebackPolicy(),
            PreferenceWritebackPolicy(),
            DecisionWritebackPolicy(),
            ExperienceWritebackPolicy(),
            SkillWritebackPolicy(),
            FailurePatternWritebackPolicy(),
            NoteWritebackPolicy(),
        ]
    )
    write_policies = [
        ProvenancePolicy(),
        TTLPolicy(TTLConfig(high_volatility_days=memory_policy.ttl.high_volatility_days, normal_days=memory_policy.ttl.normal_days)),
        PiiPolicy(),
        ConflictPolicy(reject_on_conflict=reject_conflicts),
        ConfidencePolicy(ConfidenceConfig(accept=memory_policy.confidence.accept, reject=memory_policy.confidence.reject, default_fact_confidence=1.0, reject_when_missing=False)),
        DedupPolicy(),
    ]
    repository_router = MemoryRepositoryRouter(
        vector_repo=repositories.vector,
        graph_repo=repositories.graph,
        episodic_repo=repositories.episodic,
        decision_repo=repositories.decision,
        experience_repo=repositories.experience,
        skill_repo=repositories.skill,
        failure_pattern_repo=repositories.failure_pattern,
    )
    return WriteBackService(resolver=resolver, write_policies=write_policies, repository_router=repository_router)


class OmniMemory:
    def __init__(
        self,
        use_llm: bool = False,
        distiller: ISessionMemoryDistiller | None = None,
        *,
        vector_repo: IVectorRepo | None = None,
        vector_index_backend: VectorIndexBackend | None = None,
        graph_repo: IFactRepo | None = None,
        episodic_repo: EpisodicRepo | None = None,
        decision_repo: DecisionRepo | None = None,
        experience_repo: ExperienceRepo | None = None,
        skill_repo: SkillRepo | None = None,
        failure_pattern_repo: FailurePatternRepo | None = None,
        review_queue_repo: ReviewQueueRepo | None = None,
        reject_conflicts: bool = False,
        llm: Any | None = None,
        embedder: IEmbedder | None = None,
        model_bundle: ModelBundle | None = None,
        fact_extractor: FactExtractor | None = None,
        experience_evaluator: ExperienceEvaluator | None = None,
    ) -> None:
        bundle = model_bundle or ModelBundle()
        selected_embedder = embedder or bundle.embedder
        if vector_repo is None and selected_embedder is None:
            selected_embedder = build_embedder(settings.embedding_backend, settings.embedding_model)

        self.repositories = build_memory_repositories(
            embedder=selected_embedder,
            vector_repo=vector_repo,
            vector_index_backend=vector_index_backend,
            graph_repo=graph_repo,
            episodic_repo=episodic_repo,
            decision_repo=decision_repo,
            experience_repo=experience_repo,
            skill_repo=skill_repo,
            failure_pattern_repo=failure_pattern_repo,
            review_queue_repo=review_queue_repo,
        )
        self.vector_repo = self.repositories.vector
        self.graph_repo = self.repositories.graph
        self.episodic_repo = self.repositories.episodic
        self.decision_repo = self.repositories.decision
        self.experience_repo = self.repositories.experience
        self.skill_repo = self.repositories.skill
        self.failure_pattern_repo = self.repositories.failure_pattern
        self.review_queue_repo = self.repositories.review_queue
        self.domain_graph_repo = self.repositories.domain_graph
        self.reranker = bundle.reranker

        self.retriever = Retriever(
            self.repositories.vector,
            self.repositories.graph,
            self.repositories.episodic,
            self.repositories.decision,
            self.repositories.experience,
            self.repositories.skill,
            self.repositories.failure_pattern,
            self.repositories.domain_graph,
            self.reranker,
        )
        self.consistency = SimpleConsistencyEngine()
        self.orchestrator = Orchestrator(self.retriever, self.consistency)
        self.fact_maintenance = FactMaintenanceService(self.repositories.graph)
        self.consolidator = ExperienceConsolidator(
            experience_repo=self.repositories.experience,
            skill_repo=self.repositories.skill,
            failure_pattern_repo=self.repositories.failure_pattern,
            evaluator=experience_evaluator,
        )
        self.development_cycle_recorder = DevelopmentCycleRecorder()
        self.development_memory_workflow = DevelopmentMemoryWorkflow(self)
        self.ops_memory_workflow = OpsMemoryWorkflow(self)
        self.review_queue = ReviewQueueService(repo=self.repositories.review_queue, memory=self)
        self.writeback_service = build_writeback_service(repositories=self.repositories, reject_conflicts=reject_conflicts)
        self._session_turns: list[SessionTurn] = []
        self.prompt_renderer = PromptRenderer()
        self.llm = llm if llm is not None else (bundle.llm or (build_llm() if use_llm else None))
        self.distiller = distiller or bundle.distiller or build_session_distiller(existing_llm=self.llm)
        self.fact_miner = FactMiningService(writeback_service=self.writeback_service, extractor=fact_extractor, llm=self.llm)

    def write_items(self, items: list[dict[str, Any]], *, source: str = "user", dry_run: bool = False, meta: dict[str, Any] | None = None) -> WriteReport:
        result = self.write_items_raw(items, source=source, dry_run=dry_run, meta=meta)
        return self._to_write_report(result)

    def write_items_raw(self, items: list[dict[str, Any]], *, source: str = "user", dry_run: bool = False, meta: dict[str, Any] | None = None) -> WritebackResult:
        request = WritebackRequest(items=items, source=source, dry_run=dry_run, meta=meta or {})
        with telemetry_span("memory.write_items", item_count=len(items), dry_run=dry_run):
            return self.writeback_service.write(request)
