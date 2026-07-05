from __future__ import annotations

import json
from pathlib import Path

from omni_memory.domain.models import DecisionRecord


class DecisionRepo:
    def __init__(self) -> None:
        self._store: dict[str, DecisionRecord] = {}

    def save_decision(self, decision: DecisionRecord) -> None:
        self._store[decision.id] = decision

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self._store.get(decision_id)

    def list_decisions(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[DecisionRecord]:
        decisions = list(self._store.values())
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
        for decision in self._store.values():
            haystack = _decision_text(decision)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, decision.provenance.time or 0.0, decision))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [decision for _, _, decision in scored[:k]]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = self.count()
        self._store.clear()
        return removed


class PersistentDecisionRepo:
    def __init__(self, inner: DecisionRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def save_decision(self, decision: DecisionRecord) -> None:
        self.inner.save_decision(decision)
        self._flush()

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        return self.inner.get_decision(decision_id)

    def list_decisions(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[DecisionRecord]:
        return self.inner.list_decisions(status=status, limit=limit)

    def search(self, text: str, k: int = 5) -> list[DecisionRecord]:
        return self.inner.search(text, k=k)

    def count(self) -> int:
        return self.inner.count()

    def clear(self) -> int:
        removed = self.inner.clear()
        self._flush()
        return removed

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        for item in raw:
            self.inner.save_decision(DecisionRecord.model_validate(item))

    def _flush(self) -> None:
        data = [decision.model_dump(mode="json") for decision in self.inner.list_decisions()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
