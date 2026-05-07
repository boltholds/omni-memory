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

from domain.distiller import IMemoryDistiller
from domain.models import ContextPack, ConflictReport, RetrievalBundle, WriteReport
from domain.model_ports import IEmbedder, ModelBundle
from domain.policy import MemoryPolicy
from domain.writeback import WritebackRequest, WritebackResult, WritebackRawItem
from domain.repositories import IFactRepo

from infra.consistency import SimpleConsistencyEngine
from infra.llm.llm_factory import build_llm
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo

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


def build_writeback_service(
    *,
    vector_repo: VectorStoreRepo,
    graph_repo: IFactRepo,
    episodic_repo: EpisodicRepo,
    reject_conflicts: bool = False,
) -> WriteBackService:
    memory_policy = MemoryPolicy()

    resolver = WritebackPolicyResolver(
        [
            FactWritebackPolicy(),
            EpisodeWritebackPolicy(),
            PreferenceWritebackPolicy(),
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

        self.retriever = Retriever(
            self.vector_repo,
            self.graph_repo,
            self.episodic_repo,
        )
        self.consistency = SimpleConsistencyEngine()
        self.orchestrator = Orchestrator(self.retriever, self.consistency)
        self.writeback_service = build_writeback_service(
            vector_repo=self.vector_repo,
            graph_repo=self.graph_repo,
            episodic_repo=self.episodic_repo,
            reject_conflicts=reject_conflicts,
        )

        self.distiller = distiller or bundle.distiller
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

    def retrieve(self, query: str, *, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        return self.retriever.retrieve(query, k_sem=k_sem, k_eps=k_eps)

    def build_context(self, query: str) -> ContextPack:
        bundle = self.orchestrator.plan_retrieval(query)
        return self.orchestrator.assemble_context(bundle)

    def detect_conflicts(self, query: str | None = None) -> ConflictReport:
        bundle = self.orchestrator.plan_retrieval(query or "")
        return self.consistency.detect_conflicts(bundle.facts)

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
