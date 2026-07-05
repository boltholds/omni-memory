# app/orchestrator.py
from __future__ import annotations

from typing import Any

from omni_memory.domain.models import RetrievalBundle, ContextPack
from omni_memory.domain.ports import IMemoryOrchestrator, IRetriever, IConsistencyEngine
from omni_memory.infra.metrics import metrics
from omni_memory.context_builder import build_context
from omni_memory.config import settings
from omni_memory.profiling import timed


class Orchestrator(IMemoryOrchestrator):
    def __init__(self, retriever: IRetriever, consistency: IConsistencyEngine):
        self._retriever = retriever
        self._consistency = consistency

    def plan_retrieval(
        self,
        query: str,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        metrics.inc("context_calls", 1)
        return self._retriever.retrieve(query, intent=intent, mode=mode, scope=scope)

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
