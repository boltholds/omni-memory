from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import settings
from infra.embeddings import build_embedder
from .orchestrator import Orchestrator
from .prompting import PromptRenderer
from .retriever import Retriever
from app.agent_cycle import AgentCycleRecord
from app.fact_maintenance import FactMaintenanceCommand, FactMaintenanceResult, FactMaintenanceService
from app.memory_commands import (
    MemoryCommandContext,
    MemoryCommandInterpreter,
    RecordAgentCycleCommand,
    RecordExperienceCommand,
    WriteDecisionCommand,
    WriteFactCommand,
    WriteItemsCommand,
    WriteNoteCommand,
)
from app.memory_repositories import MemoryClearCommand, MemoryClearReport, MemoryRepositories, build_memory_repositories
from domain.distiller import IMemoryDistiller, SessionTurn
from domain.models import ContextPack, ConflictReport, DecisionRecord, ExperienceRecord, RetrievalBundle, WriteReport
from domain.model_ports import IEmbedder, ModelBundle
from domain.policy import MemoryPolicy
from domain.repositories import IFactRepo
from domain.writeback import WritebackRequest, WritebackResult
from app.session_distillation import ConservativeCandidateValidator, accepted_candidates, build_transcript, candidates_to_writeback_items
from infra.consistency import SimpleConsistencyEngine
from infra.llm.llm_factory import build_llm
from infra.repo.decision_repo import DecisionRepo
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.experience_repo import ExperienceRepo
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
        distiller: IMemoryDistiller | None = None,
        *,
        vector_repo: VectorStoreRepo | None = None,
        graph_repo: IFactRepo | None = None,
        episodic_repo: EpisodicRepo | None = None,
        decision_repo: DecisionRepo | None = None,
        experience_repo: ExperienceRepo | None = None,
        reject_conflicts: bool = False,
        llm: Any | None = None,
        embedder: IEmbedder | None = None,
        model_bundle: ModelBundle | None = None,
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
        )
        self.vector_repo = self.repositories.vector
        self.graph_repo = self.repositories.graph
        self.episodic_repo = self.repositories.episodic
        self.decision_repo = self.repositories.decision
        self.experience_repo = self.repositories.experience
        self.skill_repo = self.repositories.skill
        self.failure_pattern_repo = self.repositories.failure_pattern

        self.retriever = Retriever(
            self.repositories.vector,
            self.repositories.graph,
            self.repositories.episodic,
            self.repositories.decision,
            self.repositories.experience,
            self.repositories.skill,
            self.repositories.failure_pattern,
        )
        self.consistency = SimpleConsistencyEngine()
        self.orchestrator = Orchestrator(self.retriever, self.consistency)
        self.fact_maintenance = FactMaintenanceService(self.repositories.graph)
        self.writeback_service = build_writeback_service(repositories=self.repositories, reject_conflicts=reject_conflicts)
        self.command_interpreter = MemoryCommandInterpreter(MemoryCommandContext(writeback_service=self.writeback_service))
        self.distiller = distiller or bundle.distiller
        self._session_turns: list[SessionTurn] = []
        self.reranker = bundle.reranker
        self.prompt_renderer = PromptRenderer()
        self.llm = llm if llm is not None else (bundle.llm or (build_llm() if use_llm else None))

    def write_items(self, items: list[dict[str, Any]], *, source: str = "user", dry_run: bool = False, meta: dict[str, Any] | None = None) -> WriteReport:
        result = self.write_items_raw(items, source=source, dry_run=dry_run, meta=meta)
        return self._to_write_report(result)

    def write_items_raw(self, items: list[dict[str, Any]], *, source: str = "user", dry_run: bool = False, meta: dict[str, Any] | None = None) -> WritebackResult:
        return self.command_interpreter.execute(WriteItemsCommand(items=items, source=source, dry_run=dry_run, meta=meta))

    def write_fact(self, subject: str, predicate: str, object_: str, *, source: str = "user", confidence: float = 1.0) -> WriteReport:
        return self.command_interpreter.execute(WriteFactCommand(subject=subject, predicate=predicate, object_=object_, source=source, confidence=confidence))

    def write_note(self, text: str, *, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        return self.command_interpreter.execute(WriteNoteCommand(text=text, source=source, meta=meta))

    def write_decision(self, *, title: str, decision: str, context: str = "", consequences: list[str] | None = None, alternatives: list[str] | None = None, refs: dict[str, Any] | None = None, status: str = "accepted", source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        return self.command_interpreter.execute(WriteDecisionCommand(title=title, decision=decision, context=context, consequences=consequences, alternatives=alternatives, refs=refs, status=status, source=source, meta=meta))

    def list_decisions(self, *, status: str | None = None, limit: int | None = None) -> list[DecisionRecord]:
        return self.decision_repo.list_decisions(status=status, limit=limit)

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self.decision_repo.get_decision(decision_id)

    def record_experience(self, *, goal: str, lesson: str, context: str = "", decision: str = "", actions: list[str] | None = None, outcome: str = "", evaluation: dict[str, Any] | None = None, reuse_when: list[str] | None = None, avoid_when: list[str] | None = None, confidence: float = 0.5, refs: dict[str, Any] | None = None, source: str = "user", meta: dict[str, Any] | None = None) -> WriteReport:
        return self.command_interpreter.execute(RecordExperienceCommand(goal=goal, lesson=lesson, context=context, decision=decision, actions=actions, outcome=outcome, evaluation=evaluation, reuse_when=reuse_when, avoid_when=avoid_when, confidence=confidence, refs=refs, source=source, meta=meta))

    def list_experiences(self, *, limit: int | None = None) -> list[ExperienceRecord]:
        return self.experience_repo.list_experiences(limit=limit)

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        return self.experience_repo.get_experience(experience_id)

    def search_experiences(self, query: str, *, k: int = 5) -> list[ExperienceRecord]:
        return self.experience_repo.search(query, k=k)

    def record_agent_cycle(self, cycle: AgentCycleRecord | dict[str, Any], *, source: str = "agent-cycle") -> WriteReport:
        return self.command_interpreter.execute(RecordAgentCycleCommand(cycle=cycle, source=source))

    def ingest_turn(self, role: str, content: str) -> None:
        self._session_turns.append(SessionTurn(role=role, content=content))

    def clear_session(self) -> None:
        self._session_turns.clear()

    def clear(self, *, include_vectors: bool = True, include_facts: bool = True, include_episodes: bool = True, include_decisions: bool = True, include_experiences: bool = True, include_skills: bool = True, include_failure_patterns: bool = True, include_session: bool = True, dry_run: bool = False) -> MemoryClearReport:
        command = MemoryClearCommand(include_vectors=include_vectors, include_facts=include_facts, include_episodes=include_episodes, include_decisions=include_decisions, include_experiences=include_experiences, include_skills=include_skills, include_failure_patterns=include_failure_patterns, include_session=include_session, dry_run=dry_run)
        return command.execute(self.repositories, session_turns=len(self._session_turns), clear_session=self.clear_session)

    def repository_stats(self) -> dict[str, int | None]:
        return self.repositories.stats()

    def commit_session(self, *, source: str = "session", dry_run: bool = False, meta: dict[str, Any] | None = None, min_confidence: float = 0.75, clear: bool = True) -> WritebackResult:
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
        return result

    def retrieve(self, query: str, *, k_sem: int = 5, k_eps: int = 3, intent: str | None = None, mode: str | None = None) -> RetrievalBundle:
        return self.retriever.retrieve(query, k_sem=k_sem, k_eps=k_eps, intent=intent, mode=mode)

    def build_context(self, query: str, *, intent: str | None = None, mode: str | None = None) -> ContextPack:
        bundle = self.orchestrator.plan_retrieval(query, intent=intent, mode=mode)
        return self.orchestrator.assemble_context(bundle, intent=intent, mode=mode)

    def detect_conflicts(self, query: str | None = None, *, intent: str | None = None) -> ConflictReport:
        bundle = self.orchestrator.plan_retrieval(query or "", intent=intent)
        return self.consistency.detect_conflicts(bundle.facts)

    def maintain_facts(self, command: FactMaintenanceCommand | dict[str, Any]) -> FactMaintenanceResult:
        return self.fact_maintenance.execute(command)

    def ask(self, question: str, *, lang: str = "en", style: str = "concise", temperature: float | None = None, include_context: bool = True, intent: str | None = None, mode: str | None = None) -> MemoryAnswer:
        bundle = self.orchestrator.plan_retrieval(question, intent=intent, mode=mode)
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
        context = {}
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
