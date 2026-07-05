from __future__ import annotations

from pathlib import Path

from omni_memory.domain.models import DecisionRecord
from omni_memory.infra.record_store import InMemoryRecordStoreBackend, JsonRecordStoreBackend, RecordStoreBackend


class DecisionRepo:
    def __init__(self, backend: RecordStoreBackend[DecisionRecord] | None = None) -> None:
        self._backend = backend or InMemoryRecordStoreBackend[DecisionRecord]()

    def save_decision(self, decision: DecisionRecord) -> None:
        self._backend.save(decision.id, decision)

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self._backend.get(decision_id)

    def list_decisions(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[DecisionRecord]:
        decisions = self._backend.values()
        if status:
            decisions = [decision for decision in decisions if decision.status == status]
        decisions.sort(key=lambda decision: decision.provenance.time or 0.0, reverse=True)
        if limit is not None:
            decisions = decisions[: max(0, limit)]
        return decisions

    def search(self, text: str, k: int = 5) -> list[DecisionRecord]:
        if k <= 0:
            raise ValueError("k must be > 0")
        terms = _terms(text)
        if not terms:
            return self.list_decisions(limit=k)

        scored: list[tuple[int, float, DecisionRecord]] = []
        for decision in self._backend.values():
            haystack = _decision_text(decision)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, decision.provenance.time or 0.0, decision))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [decision for _, _, decision in scored[:k]]

    def count(self) -> int:
        return self._backend.count()

    def clear(self) -> int:
        return self._backend.clear()


class PersistentDecisionRepo(DecisionRepo):
    def __init__(self, inner: DecisionRepo, path: str | Path) -> None:
        super().__init__(backend=JsonRecordStoreBackend(path, DecisionRecord))
        for item in inner.list_decisions():
            self.save_decision(item)


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in str(text or "").casefold().split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 3:
            terms.append(term)
    return terms


def _decision_text(decision: DecisionRecord) -> str:
    parts = [
        decision.title,
        decision.status,
        decision.context,
        decision.decision,
        *decision.consequences,
        *decision.alternatives,
    ]
    return " ".join(str(part or "") for part in parts).casefold()
