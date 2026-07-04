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
from app.ops_cycle import OpsCycleDraft
from app.ops_memory_workflow import OpsMemoryWorkflow
from app.review_queue import ReviewActionResult, ReviewQueueService
from domain.distiller import ISessionMemoryDistiller, SessionTurn
from domain.experience_evaluator import ExperienceEvaluator
from domain.models import ContextPack, ConflictReport, DecisionRecord, ExperienceRecord, RetrievalBundle, ReviewItem, WriteReport
from domain.model_ports import IEmbedder, ModelBundle
from domain.policy import MemoryPolicy
from domain.repositories import IFactRepo
from domain.writeback import WritebackRequest, WritebackResult
from app.session_distillation import ConservativeCandidateValidator, accepted_candidates, build_transcript, candidates_to_writeback_items
from app.telemetry import span as telemetry_span
from infra.consistency import SimpleConsistencyEngine
from infra.distillers.factory import build_session_distiller
from infra.llm.llm_factory import build_llm
from infra.repo.decision_repo import DecisionRepo
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.experience_repo import ExperienceRepo
from infra.repo.cognitive_repo import FailurePatternRepo, SkillRepo
from infra.repo.review_repo import ReviewQueueRepo
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
        with telemetry_span("memory.write", item_count=len(items), source=source, dry_run=dry_run) as span:
            result = self.writeback_service.write(WriteItemsCommand(items=items, source=source, dry_run=dry_run, meta=meta).to_request())
            _set_span_result_counts(span, result)
            return result

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
        return self.development_memory_workflow.finish_task(
            {
                "goal": task_id or "Finish development task",
                "lesson": lesson or "Review this development task and fill a reusable lesson before recording.",
                "changed_files": changed_files or [],
                "commands_run": commands_run or [],
                "tests": tests or [],
                "decisions": decisions or [],
                "outcome": outcome,
                "side_effects": side_effects or [],
                "confidence": confidence,
                "source": source,
                "run_distiller": False,
            }
        )

    def record_agent_cycle(self, cycle: AgentCycleRecord | dict[str, Any], *, source: str = "agent-cycle") -> WriteReport:
        item = RecordAgentCycleCommand(cycle=cycle, source=source).to_item()
        return self._to_write_report(self._write_item_raw(item, source=source))

    def draft_development_cycle(self, cycle: DevelopmentCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        return self.development_cycle_recorder.draft(cycle)

    def record_development_cycle(self, cycle: DevelopmentCycleDraft | dict[str, Any], *, source: str = "development-cycle") -> WriteReport:
        draft = self.draft_development_cycle(cycle)
        return self.record_agent_cycle(draft, source=source)

    def draft_ops_cycle(self, cycle: OpsCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        return self.ops_memory_workflow.draft_cycle(cycle)

    def record_ops_cycle(self, cycle: OpsCycleDraft | dict[str, Any], *, source: str = "ops-workflow") -> WriteReport:
        return self.ops_memory_workflow.record_cycle(cycle, source=source)

    def consolidate_experiences(self, *, dry_run: bool = True, min_confidence: float = 0.85) -> ConsolidationResult:
        with telemetry_span("memory.consolidate", dry_run=dry_run, min_confidence=min_confidence) as span:
            result = self.consolidator.consolidate(dry_run=dry_run, min_confidence=min_confidence)
            _set_span_attribute(span, "proposal_count", len(result.proposals))
            _set_span_attribute(span, "saved_skill_count", len(result.saved_skills))
            _set_span_attribute(span, "saved_failure_pattern_count", len(result.saved_failure_patterns))
            return result

    def submit_review_item(self, *, kind: str, title: str, payload: dict[str, Any], confidence: float = 0.5, reason: str = "", source: str = "review-queue", meta: dict[str, Any] | None = None) -> ReviewItem:
        return self.review_queue.submit(kind=kind, title=title, payload=payload, confidence=confidence, reason=reason, source=source, meta=meta)

    def list_review_items(self, *, status: str | None = None, kind: str | None = None, limit: int | None = None) -> list[ReviewItem]:
        return self.review_queue.list(status=status, kind=kind, limit=limit)

    def get_review_item(self, item_id: str) -> ReviewItem | None:
        return self.review_queue.get(item_id)

    def accept_review_item(self, item_id: str, *, reviewer: str = "user", note: str = "") -> ReviewActionResult:
        return self.review_queue.accept(item_id, reviewer=reviewer, note=note)

    def reject_review_item(self, item_id: str, *, reviewer: str = "user", note: str = "") -> ReviewActionResult:
        return self.review_queue.reject(item_id, reviewer=reviewer, note=note)

    def supersede_review_item(self, item_id: str, *, replacement: dict[str, Any], reviewer: str = "user", note: str = "") -> ReviewActionResult:
        return self.review_queue.supersede(item_id, replacement=replacement, reviewer=reviewer, note=note)

    def ingest_turn(self, role: str, content: str) -> None:
        self._session_turns.append(SessionTurn(role=role, content=content))

    def clear_session(self) -> None:
        self._session_turns.clear()

    def clear(self, *, include_vectors: bool = True, include_facts: bool = True, include_episodes: bool = True, include_decisions: bool = True, include_experiences: bool = True, include_skills: bool = True, include_failure_patterns: bool = True, include_review_items: bool = True, include_session: bool = True, dry_run: bool = False) -> MemoryClearReport:
        command = MemoryClearCommand(include_vectors=include_vectors, include_facts=include_facts, include_episodes=include_episodes, include_decisions=include_decisions, include_experiences=include_experiences, include_skills=include_skills, include_failure_patterns=include_failure_patterns, include_review_items=include_review_items, include_session=include_session, dry_run=dry_run)
        return command.execute(self.repositories, session_turns=len(self._session_turns), clear_session=self.clear_session)

    def repository_stats(self) -> dict[str, int | None]:
        return self.repositories.stats()

    def commit_session(self, *, source: str = "session", dry_run: bool = False, meta: dict[str, Any] | None = None, min_confidence: float = 0.75, clear: bool = True) -> WritebackResult:
        with telemetry_span("memory.distill", source=source, dry_run=dry_run, turn_count=len(self._session_turns), min_confidence=min_confidence) as span:
            if not self._session_turns:
                return WritebackResult()
            if self.distiller is None or not hasattr(self.distiller, "distill_session"):
                raise RuntimeError("Session distiller is not configured. Pass a distiller with distill_session().")
            turns = list(self._session_turns)
            transcript = build_transcript(turns)
            distillation = self.distiller.distill_session(turns)
            candidates, rejected = accepted_candidates(distillation, transcript=transcript, validator=ConservativeCandidateValidator(min_confidence=min_confidence))
            raw_items = candidates_to_writeback_items(candidates, source=source, meta=meta or {"session_rejected": rejected})
            result = self.writeback_service.write(WritebackRequest(source=source, dry_run=dry_run, meta=meta or {}, items=raw_items))
            for reason in rejected:
                from domain.writeback import WritebackDecision
                result.add_rejected(WritebackDecision.reject(reason=reason, policy="SessionDistillation"))
            if clear and not dry_run:
                self.clear_session()
            _set_span_attribute(span, "candidate_count", len(candidates))
            _set_span_attribute(span, "rejected_candidate_count", len(rejected))
            _set_span_result_counts(span, result)
            return result

    def mine_facts(self, text: str, *, source: str = "fact-mining", dry_run: bool = True, min_confidence: float = 0.75, policy_mode: str = "review", domain_ids: list[str] | None = None, meta: dict[str, Any] | None = None, extractor: FactExtractor | None = None) -> FactMiningResult:
        return self.fact_miner.mine_text(
            text,
            source=source,
            dry_run=dry_run,
            min_confidence=min_confidence,
            policy_mode=policy_mode,  # type: ignore[arg-type]
            domain_ids=domain_ids or [],
            meta=meta or {},
            extractor=extractor,
        )

    def retrieve(self, query: str, *, k_sem: int = 5, k_eps: int = 3, intent: str | None = None, mode: str | None = None, scope: dict[str, Any] | None = None) -> RetrievalBundle:
        with telemetry_span("memory.retrieve", k_sem=k_sem, k_eps=k_eps, intent=intent, mode=mode, scoped=scope is not None) as span:
            bundle = self.retriever.retrieve(query, k_sem=k_sem, k_eps=k_eps, intent=intent, mode=mode, scope=scope)
            _set_span_attribute(span, "semantic_count", len(bundle.semantic_chunks))
            _set_span_attribute(span, "fact_count", len(bundle.facts))
            _set_span_attribute(span, "episode_count", len(bundle.episodes))
            _set_span_attribute(span, "experience_count", len(bundle.experiences))
            _set_span_attribute(span, "skill_count", len(bundle.skills))
            _set_span_attribute(span, "failure_pattern_count", len(bundle.failure_patterns))
            return bundle

    def build_context(self, query: str, *, intent: str | None = None, mode: str | None = None, scope: dict[str, Any] | None = None) -> ContextPack:
        bundle = self.orchestrator.plan_retrieval(query, intent=intent, mode=mode, scope=scope)
        return self.orchestrator.assemble_context(bundle, intent=intent, mode=mode)

    def detect_conflicts(self, query: str | None = None, *, intent: str | None = None, scope: dict[str, Any] | None = None) -> ConflictReport:
        bundle = self.orchestrator.plan_retrieval(query or "", intent=intent, scope=scope)
        return self.consistency.detect_conflicts(bundle.facts)

    def maintain_facts(self, command: FactMaintenanceCommand | dict[str, Any]) -> FactMaintenanceResult:
        return self.fact_maintenance.execute(command)

    def ask(self, question: str, *, lang: str = "en", style: str = "concise", temperature: float | None = None, include_context: bool = True, intent: str | None = None, mode: str | None = None, scope: dict[str, Any] | None = None) -> MemoryAnswer:
        bundle = self.orchestrator.plan_retrieval(question, intent=intent, mode=mode, scope=scope)
        pack = self.orchestrator.assemble_context(bundle, intent=intent, mode=mode)
        conflict_report = self.consistency.detect_conflicts(bundle.facts)
        if self.llm is None:
            answer = "LLM provider is not configured. Use retrieve/build_context or pass use_llm=True."
            model = None
        else:
            sections = [f"{s.title}:\n{s.body}" for s in pack.sections]
            messages = self.prompt_renderer.make_messages(question, sections, lang=lang, style=style)
            result = self.llm.generate(messages, temperature=temperature)
            answer = result.get("text", "").strip()
            model = result.get("model")
        context: dict[str, Any] = {}
        if include_context:
            context = {
                "facts": [f.model_dump() for f in bundle.facts],
                "episodes": [e.model_dump() for e in bundle.episodes],
                "decisions": [d.model_dump() for d in bundle.decisions],
                "experiences": [e.model_dump() for e in bundle.experiences],
                "skills": [s.model_dump() for s in bundle.skills],
                "failure_patterns": [p.model_dump() for p in bundle.failure_patterns],
                "semantic_chunks": [s.model_dump() for s in bundle.semantic_chunks],
                "conflicts": [c.model_dump() for c in conflict_report.conflicts],
                "sections": [s.model_dump() for s in pack.sections],
            }
        return MemoryAnswer(answer=answer, advisories=pack.advisories, used_sections=[s.title for s in pack.sections], context=context, model=model)

    @staticmethod
    def _to_write_report(result: WritebackResult) -> WriteReport:
        return WriteReport(saved=result.saved_count, rejected=result.rejected_count + result.error_count, reasons=result.reasons)


def _set_span_result_counts(span: Any | None, result: WritebackResult) -> None:
    _set_span_attribute(span, "saved_count", result.saved_count)
    _set_span_attribute(span, "rejected_count", result.rejected_count)
    _set_span_attribute(span, "error_count", result.error_count)


def _set_span_attribute(span: Any | None, key: str, value: Any) -> None:
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute(key, value)
