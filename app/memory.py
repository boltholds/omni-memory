from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import settings
from infra.embeddings import build_embedder
from .orchestrator import Orchestrator
from .prompting import PromptRenderer
from .retriever import Retriever
from app.agent_cycle import AgentCycleRecord
from app.consolidation import ConsolidationResult, ExperienceConsolidator
from app.development_cycle import DevelopmentCycleDraft, DevelopmentCycleRecorder
from app.development_memory_workflow import DevelopmentMemoryWorkflow, FinishDevelopmentTaskResult
from app.fact_maintenance import FactMaintenanceCommand, FactMaintenanceResult, FactMaintenanceService
from app.fact_mining import FactExtractor, FactMiningResult, FactMiningService
from app.memory_commands import (
    RecordAgentCycleCommand,
    RecordExperienceCommand,
    WriteFailurePatternCommand,
    WriteDecisionCommand,
    WriteFactCommand,
    WriteItemsCommand,
    WriteNoteCommand,
    WriteSkillCommand,
)
from app.memory_repositories import MemoryClearCommand, MemoryClearReport, MemoryRepositories, build_memory_repositories
from domain.distiller import ISessionMemoryDistiller, SessionTurn
from domain.experience_evaluator import ExperienceEvaluator
from domain.models import ContextPack, ConflictReport, DecisionRecord, ExperienceRecord, RetrievalBundle, WriteReport
from domain.model_ports import IEmbedder, ModelBundle
from domain.policy import MemoryPolicy
from domain.repositories import IFactRepo
from domain.writeback import WritebackRequest, WritebackResult
from app.session_distillation import ConservativeCandidateValidator, accepted_candidates, build_transcript, candidates_to_writeback_items
from infra.consistency import SimpleConsistencyEngine
from infra.distillers.factory import build_session_distiller
from infra.llm.llm_factory import build_llm
from infra.repo.decision_repo import DecisionRepo
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.experience_repo import ExperienceRepo
from infra.repo.cognitive_repo import FailurePatternRepo, SkillRepo
from infra.repo.vector_repo import VectorStoreRepo
from app.writeback.memory_policies import ConfidenceConfig, ConfidencePolicy, ConflictPolicy, DedupPolicy, PiiPolicy, ProvenancePolicy, TTLConfig, TTLPolicy
from app.writeback.service import MemoryRepositoryRouter, WriteBackService
from app.writeback.writeback_policies import (
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
        vector_repo: VectorStoreRepo | None = None,
        graph_repo: IFactRepo | None = None,
        episodic_repo: EpisodicRepo | None = None,
        decision_repo: DecisionRepo | None = None,
        experience_repo: ExperienceRepo | None = None,
        skill_repo: SkillRepo | None = None,
        failure_pattern_repo: FailurePatternRepo | None = None,
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
            graph_repo=graph_repo,
            episodic_repo=episodic_repo,
            decision_repo=decision_repo,
            experience_repo=experience_repo,
            skill_repo=skill_repo,
            failure_pattern_repo=failure_pattern_repo,
        )
        self.vector_repo = self.repositories.vector
        self.graph_repo = self.repositories.graph
        self.episodic_repo = self.repositories.episodic
        self.decision_repo = self.repositories.decision
        self.experience_repo = self.repositories.experience
        self.skill_repo = self.repositories.skill
        self.failure_pattern_repo = self.repositories.failure_pattern
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
        return self.writeback_service.write(WriteItemsCommand(items=items, source=source, dry_run=dry_run, meta=meta).to_request())

    def _write_item_raw(self, item: dict[str, Any], *, source: str, meta: dict[str, Any] | None = None, dry_run: bool = False) -> WritebackResult:
        return self.writeback_service.write(WriteItemsCommand(items=[item], source=source, dry_run=dry_run, meta=meta).to_request())

    def write_fact(self, subject: str, predicate: str, object_: str, *, source: str = "user", confidence: float = 1.0) -> WriteReport:
        item = WriteFactCommand(subject=subject, predicate=predicate, object_=object_, source=source, confidence=confidence).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source))

    def write_note(self, text: str, *, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        item = WriteNoteCommand(text=text, source=source, meta=meta).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source, meta=meta))

    def write_decision(self, *, title: str, decision: str, context: str = "", consequences: list[str] | None = None, alternatives: list[str] | None = None, refs: dict[str, Any] | None = None, status: str = "accepted", source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        item = WriteDecisionCommand(title=title, decision=decision, context=context, consequences=consequences, alternatives=alternatives, refs=refs, status=status, source=source, meta=meta).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source, meta=meta))

    def list_decisions(self, *, status: str | None = None, limit: int | None = None) -> list[DecisionRecord]:
        return self.decision_repo.list_decisions(status=status, limit=limit)

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self.decision_repo.get_decision(decision_id)

    def record_experience(self, *, goal: str, lesson: str, context: str = "", decision: str = "", actions: list[str] | None = None, outcome: str = "", evaluation: dict[str, Any] | None = None, reuse_when: list[str] | None = None, avoid_when: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        item = RecordExperienceCommand(goal=goal, lesson=lesson, context=context, decision=decision, actions=actions, outcome=outcome, evaluation=evaluation, reuse_when=reuse_when, avoid_when=avoid_when, confidence=confidence, refs=refs, source=source, meta=meta).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source, meta=meta))

    def write_skill(self, *, name: str, problem: str = "", procedure: list[str] | None = None, reuse_when: list[str] | None = None, avoid_when: list[str] | None = None, evidence_ids: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        result = self.write_skill_raw(name=name, problem=problem, procedure=procedure, reuse_when=reuse_when, avoid_when=avoid_when, evidence_ids=evidence_ids, confidence=confidence, refs=refs, source=source, meta=meta)
        return self._to_write_report(result)

    def write_skill_raw(self, *, name: str, problem: str = "", procedure: list[str] | None = None, reuse_when: list[str] | None = None, avoid_when: list[str] | None = None, evidence_ids: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WritebackResult:
        item = WriteSkillCommand(name=name, problem=problem, procedure=procedure, reuse_when=reuse_when, avoid_when=avoid_when, evidence_ids=evidence_ids, confidence=confidence, refs=refs, source=source, meta=meta).to_item()
        return self._write_item_raw(item, source=source, meta=meta)

    def write_failure_pattern(self, *, symptom: str, root_cause: str = "", fix: str = "", detection: str = "", evidence_ids: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        result = self.write_failure_pattern_raw(symptom=symptom, root_cause=root_cause, fix=fix, detection=detection, evidence_ids=evidence_ids, confidence=confidence, refs=refs, source=source, meta=meta)
        return self._to_write_report(result)

    def write_failure_pattern_raw(self, *, symptom: str, root_cause: str = "", fix: str = "", detection: str = "", evidence_ids: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WritebackResult:
        item = WriteFailurePatternCommand(symptom=symptom, root_cause=root_cause, fix=fix, detection=detection, evidence_ids=evidence_ids, confidence=confidence, refs=refs, source=source, meta=meta).to_item()
        return self._write_item_raw(item, source=source, meta=meta)

    def list_experiences(self, *, limit: int | None = None) -> list[ExperienceRecord]:
        return self.experience_repo.list_experiences(limit=limit)

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        return self.experience_repo.get_experience(experience_id)

    def search_experiences(self, query: str, *, k: int = 5) -> list[ExperienceRecord]:
        return self.experience_repo.search(query, k=k)

    def start_development_task(self, *, goal: str, context: str = "", constraints: list[str] | None = None, files: list[str] | None = None, source: str = "development-workflow"):
        return self.development_memory_workflow.start_task(goal=goal, context=context, constraints=constraints, files=files, source=source)

    def finish_development_task(self, *, task_id: str | None = None, outcome: str, tests: list[str] | None = None, changed_files: list[str] | None = None, lesson: str = "", decisions: list[str] | None = None, commands_run: list[str] | None = None, side_effects: list[str] | None = None, confidence: float = 0.8, source: str = "development-workflow") -> FinishDevelopmentTaskResult:
        return self.development_memory_workflow.finish_task(task_id=task_id, outcome=outcome, tests=tests, changed_files=changed_files, lesson=lesson, decisions=decisions, commands_run=commands_run, side_effects=side_effects, confidence=confidence, source=source)

    def record_agent_cycle(self, cycle: AgentCycleRecord | dict[str, Any], *, source: str = "agent-cycle") -> WriteReport:
        item = RecordAgentCycleCommand(cycle=cycle, source=source).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source))

    def draft_development_cycle(self, cycle: DevelopmentCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        return self.development_cycle_recorder.draft(cycle)

    def record_development_cycle(self, cycle: DevelopmentCycleDraft | dict[str, Any], *, source: str = "development-cycle") -> WriteReport:
        draft = self.draft_development_cycle(cycle)
        return self.record_agent_cycle(draft, source=source)

    def consolidate_experiences(self, *, dry_run: bool = True, min_confidence: float = 0.85) -> ConsolidationResult:
        return self.consolidator.consolidate(dry_run=dry_run, min_confidence=min_confidence)

    def ingest_turn(self, role: str, content: str) -> None:
        self._session_turns.append(SessionTurn(role=role, content=content))
