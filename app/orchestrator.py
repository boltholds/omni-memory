# app/orchestrator.py
from __future__ import annotations
from typing import List

from domain.models import RetrievalBundle, ContextPack, ContextSection, ConflictReport
from domain.ports import IMemoryOrchestrator, IRetriever, IConsistencyEngine
from infra.metrics import metrics

class Orchestrator(IMemoryOrchestrator):
    def __init__(self, retriever: IRetriever, consistency: IConsistencyEngine):
        self._retriever = retriever
        self._consistency = consistency

    def plan_retrieval(self, query: str) -> RetrievalBundle:
        metrics.inc("context_calls", 1)
        return self._retriever.retrieve(query)

    def assemble_context(self, bundle: RetrievalBundle) -> ContextPack:
        sections: List[ContextSection] = []

        # --- семантические куски ---
        if bundle.semantic_chunks:
            body = "\n".join(
                [f"- {o.payload.get('text') or str(o.payload)}" for o in bundle.semantic_chunks]
            )
            sections.append(ContextSection(title="Semantic Notes", body=body))

        # --- факты + конфликты ---
        if bundle.facts:
            facts_str = "\n".join([f"- {f.subject} {f.predicate} {f.object}" for f in bundle.facts])
            sections.append(ContextSection(title="Facts", body=facts_str))

            report: ConflictReport = self._consistency.detect_conflicts(bundle.facts)
            if report.conflicts:
                metrics.inc("conflicts_detected", len(report.conflicts))
                body = "\n".join([f"- {c.key}: {', '.join(c.variants)}" for c in report.conflicts])
                sections.append(ContextSection(title="Conflicts", body=body))

        # --- эпизоды ---
        if bundle.episodes:
            epis_str = "\n".join([f"- {e.summary}" for e in bundle.episodes])
            sections.append(ContextSection(title="Episodes", body=epis_str))

        return ContextPack(sections=sections, advisories=[])
