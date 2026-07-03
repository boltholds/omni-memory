# app/orchestrator.py
from __future__ import annotations

from domain.models import RetrievalBundle, ContextPack
from domain.ports import IMemoryOrchestrator, IRetriever, IConsistencyEngine
from infra.metrics import metrics
from app.context_builder import build_context
from app.config import settings
from app.profiling import timed


class Orchestrator(IMemoryOrchestrator):
    def __init__(self, retriever: IRetriever, consistency: IConsistencyEngine):
        self._retriever = retriever
        self._consistency = consistency

    def plan_retrieval(self, query: str, intent: str | None = None, mode: str | None = None) -> RetrievalBundle:
        metrics.inc("context_calls", 1)
        if intent is None and mode is None:
            return self._retriever.retrieve(query)
        return self._retriever.retrieve(query, intent=intent, mode=mode)

    @timed("retriever.retrieve", slow_ms=100)
    def assemble_context(self, bundle: RetrievalBundle, intent: str | None = None, mode: str | None = None) -> ContextPack:
        pack, advisories = build_context(
            bundle,
            settings.context_max_tokens,
            consistency=self._consistency,
            intent=intent or mode,
        )
        pack.advisories = list(dict.fromkeys(advisories))
        return pack
