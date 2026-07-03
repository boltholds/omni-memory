from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .config import settings
from infra.embeddings import build_embedder
from .orchestrator import Orchestrator
from .prompting import PromptRenderer
from .retriever import Retriever
from app.fact_maintenance import FactMaintenanceCommand, FactMaintenanceResult, FactMaintenanceService

from domain.distiller import IMemoryDistiller, SessionTurn
from domain.models import ContextPack, ConflictReport, DecisionRecord, RetrievalBundle, WriteReport
from domain.model_ports import IEmbedder, ModelBundle
from domain.policy import MemoryPolicy
from domain.writeback import WritebackRequest, WritebackResult, WritebackRawItem
from app.session_distillation import (
    ConservativeCandidateValidator,
    accepted_candidates,
    build_transcript,
    candidates_to_writeback_items,
)
from domain.repositories import IFactRepo

from infra.consistency import SimpleConsistencyEngine
from infra.llm.llm_factory import build_llm
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo
from infra.repo.decision_repo import DecisionRepo

from app.writeback.memory_policies import (
    ConfidenceConfig,
    ConfidencePolicy,
    ConflictPolicy,
    DedupPolicy,
    PiiPolicy,
    ProvenancePolicy,
    TTLConfig,
    TTLPolicy,
)
from app.writeback.service import MemoryRepositoryRouter, WriteBackService
from app.writeback.writeback_policies import (
    DecisionWritebackPolicy,
    EpisodeWritebackPolicy,
    FactWritebackPolicy,
    NoteWritebackPolicy,
    PreferenceWritebackPolicy,
    WritebackPolicyResolver,
)


@dataclass(frozen=True)
class MemoryAnswer:
    answer: str
    advisories: list[str]
    used_sections: list[str]
    context: dict[str, Any]
    model: str | None = None


@dataclass(frozen=True)
class MemoryClearReport:
    vector_objects: int = 0
    facts: int = 0
    episodes: int = 0
    decisions: int = 0
    session_turns: int = 0
    dry_run: bool = False


def build_writeback_service(
    *,
    vector_repo: VectorStoreRepo,
    graph_repo: IFactRepo,
    episodic_repo: EpisodicRepo,
    decision_repo: DecisionRepo | None = None,
    reject_conflicts: bool = False,
) -> WriteBackService:
    memory_policy = MemoryPolicy()

    resolver = WritebackPolicyResolver(
        [
            FactWritebackPolicy(),
            EpisodeWritebackPolicy(),
            PreferenceWritebackPolicy(),
            DecisionWritebackPolicy(),
            NoteWritebackPolicy(),
        ]
    )

    write_policies = [
        ProvenancePolicy(),
        TTLPolicy(
            TTLConfig(
                high_volatility_days=memory_policy.ttl.high_volatility_days,
                normal_days=memory_policy.ttl.normal_days,
            )
        ),
        PiiPolicy(),
        ConflictPolicy(reject_on_conflict=reject_conflicts),
        ConfidencePolicy(
            ConfidenceConfig(
                accept=memory_policy.confidence.accept,
                reject=memory_policy.confidence.reject,
                default_fact_confidence=1.0,
                reject_when_missing=False,
            )
        ),
        DedupPolicy(),
    ]

    repository_router = MemoryRepositoryRouter(
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        episodic_repo=episodic_repo,
        decision_repo=decision_repo,
    )

    return WriteBackService(
        resolver=resolver,
        write_policies=write_policies,
        repository_router=repository_router,
    )


class OmniMemory:
    """
    Public facade for memory operations.

    CLI, FastAPI, MCP servers and LangGraph nodes should depend on this class,
    not on repositories, writeback policies or demo-specific helpers.
    """

    def __init__(
        self,
        use_llm: bool = False,
        distiller: IMemoryDistiller | None = None,
        *,
        vector_repo: VectorStoreRepo | None = None,
        graph_repo: IFactRepo | None = None,
        episodic_repo: EpisodicRepo | None = None,
        decision_repo: DecisionRepo | None = None,
        reject_conflicts: bool = False,
        llm: Any | None = None,
        embedder: IEmbedder | None = None,
        model_bundle: ModelBundle | None = None,
    ) -> None:
        bundle = model_bundle or ModelBundle()

        selected_embedder = embedder or bundle.embedder
        if vector_repo is None and selected_embedder is None:
            selected_embedder = build_embedder(settings.embedding_backend, settings.embedding_model)

        self.vector_repo = vector_repo or VectorStoreRepo(embedder=selected_embedder)
        self.graph_repo = graph_repo or GraphRepo()
        self.episodic_repo = episodic_repo or EpisodicRepo(db_path=settings.sqlite_path)
        self.decision_repo = decision_repo or DecisionRepo()

        self.retriever = Retriever(
            self.vector_repo,
            self.graph_repo,
            self.episodic_repo,
            self.decision_repo,
        )
        self.consistency = SimpleConsistencyEngine()
        self.orchestrator = Orchestrator(self.retriever, self.consistency)
        self.fact_maintenance = FactMaintenanceService(self.graph_repo)
        self.writeback_service = build_writeback_service(
            vector_repo=self.vector_repo,
            graph_repo=self.graph_repo,
            episodic_repo=self.episodic_repo,
            decision_repo=self.decision_repo,
            reject_conflicts=reject_conflicts,
        )

        self.distiller = distiller or bundle.distiller
        self._session_turns: list[SessionTurn] = []
        self.reranker = bundle.reranker
        self.prompt_renderer = PromptRenderer()
        self.llm = llm if llm is not None else (bundle.llm or (build_llm() if use_llm else None))

    def write_items(
        self,
        items: list[dict[str, Any]],
        *,
        source: str = "user",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> WriteReport:
        result = self.write_items_raw(
            items,
            source=source,
            dry_run=dry_run,
            meta=meta,
        )
        return self._to_write_report(result)

    def write_items_raw(
        self,
        items: list[dict[str, Any]],
        *,
        source: str = "user",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> WritebackResult:
        request = WritebackRequest(
                source=source,
                dry_run= dry_run,
                meta= meta or {},
                items=[WritebackRawItem.model_validate(item) for item in items],
        )
        return self.writeback_service.write(request)

    def write_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        source: str = "user",
        confidence: float = 1.0,
    ) -> WriteReport:
        item = {
            "id": f"fact-{uuid.uuid4().hex}",
            "type": "fact",
            "subject": subject.lower().strip(),
            "predicate": predicate.lower().strip(),
            "object": object_.lower().strip(),
            "provenance": {
                "source": source,
                "time": time.time(),
                "meta": {},
            },
            "meta": {
                "confidence": confidence,
            },
        }
        return self.write_items([item], source=source)

    def write_note(
        self,
        text: str,
        *,
        source: str = "user",
        meta: dict[str, Any] | None = None,
    ) -> WriteReport:
        item = {
            "id": f"note-{uuid.uuid4().hex}",
            "type": "note",
            "payload": {"text": text},
            "provenance": {
                "source": source,
                "time": time.time(),
                "meta": meta or {},
            },
            "meta": meta or {},
        }
        return self.write_items([item], source=source, meta=meta)

    def write_decision(
        self,
        *,
        title: str,
        decision: str,
        context: str = "",
        consequences: list[str] | None = None,
        alternatives: list[str] | None = None,
        refs: dict[str, Any] | None = None,
        status: str = "accepted",
        source: str = "user",
        meta: dict[str, Any] | None = None,
    ) -> WriteReport:
        item = {
            "id": f"decision-{uuid.uuid4().hex}",
            "type": "decision",
            "payload": {
                "title": title,
                "status": status,
                "context": context,
                "decision": decision,
                "consequences": consequences or [],
                "alternatives": alternatives or [],
                "refs": refs or {},
            },
            "provenance": {
                "source": source,
                "time": time.time(),
                "meta": {},
            },
            "meta": meta or {},
        }
        return self.write_items([item], source=source, meta=meta)

    def list_decisions(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[DecisionRecord]:
        return self.decision_repo.list_decisions(status=status, limit=limit)

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self.decision_repo.get_decision(decision_id)

    def ingest_turn(self, role: str, content: str) -> None:
        self._session_turns.append(SessionTurn(role=role, content=content))

    def clear_session(self) -> None:
        self._session_turns.clear()

    def clear(
        self,
        *,
        include_vectors: bool = True,
        include_facts: bool = True,
        include_episodes: bool = True,
        include_decisions: bool = True,
        include_session: bool = True,
        dry_run: bool = False,
    ) -> MemoryClearReport:
        vector_objects = _repo_count(self.vector_repo) if include_vectors else 0
        facts = _repo_count(self.graph_repo) if include_facts else 0
        episodes = _repo_count(self.episodic_repo) if include_episodes else 0
        decisions = _repo_count(self.decision_repo) if include_decisions else 0
        session_turns = len(self._session_turns) if include_session else 0

        if not dry_run:
            if include_vectors:
                _repo_clear(self.vector_repo)
            if include_facts:
                _repo_clear(self.graph_repo)
            if include_episodes:
                _repo_clear(self.episodic_repo)
            if include_decisions:
                _repo_clear(self.decision_repo)
            if include_session:
                self.clear_session()

        return MemoryClearReport(
            vector_objects=vector_objects,
            facts=facts,
            episodes=episodes,
            decisions=decisions,
            session_turns=session_turns,
            dry_run=dry_run,
        )

    def commit_session(
        self,
        *,
        source: str = "session",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
        min_confidence: float = 0.75,
        clear: bool = True,
    ) -> WritebackResult:
        """Distill the accumulated dialogue session into durable memory.

        This is intentionally conservative: the distiller proposes candidates,
        then the validator filters them before regular writeback policies run.
        """

        if not self._session_turns:
            return WritebackResult()

        if self.distiller is None or not hasattr(self.distiller, "distill_session"):
            raise RuntimeError("Session distiller is not configured. Pass a distiller with distill_session().")

        turns = list(self._session_turns)
        transcript = build_transcript(turns)
        distillation = self.distiller.distill_session(turns)
        candidates, rejected = accepted_candidates(
            distillation,
            transcript=transcript,
            validator=ConservativeCandidateValidator(min_confidence=min_confidence),
        )
        raw_items = candidates_to_writeback_items(
            candidates,
            source=source,
            meta=meta or {"session_rejected": rejected},
        )
        result = self.writeback_service.write(
            WritebackRequest(
                source=source,
                dry_run=dry_run,
                meta=meta or {},
                items=raw_items,
            )
        )

        for reason in rejected:
            from domain.writeback import WritebackDecision

            result.add_rejected(WritebackDecision.reject(reason=reason, policy="SessionDistillation"))

        if clear and not dry_run:
            self.clear_session()

        return result

    def retrieve(self, query: str, *, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        return self.retriever.retrieve(query, k_sem=k_sem, k_eps=k_eps)

    def build_context(self, query: str) -> ContextPack:
        bundle = self.orchestrator.plan_retrieval(query)
        return self.orchestrator.assemble_context(bundle)

    def detect_conflicts(self, query: str | None = None) -> ConflictReport:
        bundle = self.orchestrator.plan_retrieval(query or "")
        return self.consistency.detect_conflicts(bundle.facts)

    def maintain_facts(
        self,
        command: FactMaintenanceCommand | dict[str, Any],
    ) -> FactMaintenanceResult:
        return self.fact_maintenance.execute(command)

    def ask(
        self,
        question: str,
        *,
        lang: str = "en",
        style: str = "concise",
        temperature: float | None = None,
        include_context: bool = True,
    ) -> MemoryAnswer:
        bundle = self.orchestrator.plan_retrieval(question)
        pack = self.orchestrator.assemble_context(bundle)
        conflict_report = self.consistency.detect_conflicts(bundle.facts)

        if self.llm is None:
            answer = "LLM provider is not configured. Use retrieve/build_context or pass use_llm=True."
            model = None
        else:
            sections = [f"{s.title}:\n{s.body}" for s in pack.sections]
            messages = self.prompt_renderer.make_messages(
                question,
                sections,
                lang=lang,
                style=style,
            )
            result = self.llm.generate(messages, temperature=temperature)
            answer = result.get("text", "").strip()
            model = result.get("model")

        context = {}
        if include_context:
            context = {
                "facts": [f.model_dump() for f in bundle.facts],
                "episodes": [e.model_dump() for e in bundle.episodes],
                "decisions": [d.model_dump() for d in bundle.decisions],
                "semantic_chunks": [s.model_dump() for s in bundle.semantic_chunks],
                "conflicts": [c.model_dump() for c in conflict_report.conflicts],
                "sections": [s.model_dump() for s in pack.sections],
            }

        return MemoryAnswer(
            answer=answer,
            advisories=pack.advisories,
            used_sections=[s.title for s in pack.sections],
            context=context,
            model=model,
        )

    @staticmethod
    def _to_write_report(result: WritebackResult) -> WriteReport:
        return WriteReport(
            saved=result.saved_count,
            rejected=result.rejected_count + result.error_count,
            reasons=result.reasons,
        )


def _repo_count(repo: Any) -> int:
    return int(repo.count()) if hasattr(repo, "count") else 0


def _repo_clear(repo: Any) -> int:
    if not hasattr(repo, "clear"):
        return 0
    return int(repo.clear())
