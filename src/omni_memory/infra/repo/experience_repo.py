from __future__ import annotations

import json
from pathlib import Path

from omni_memory.domain.models import ExperienceRecord
from omni_memory.infra.record_store import InMemoryRecordStoreBackend, JsonRecordStoreBackend, RecordStoreBackend


class ExperienceRepo:
    def __init__(self, backend: RecordStoreBackend[ExperienceRecord] | None = None) -> None:
        self._backend = backend or InMemoryRecordStoreBackend[ExperienceRecord]()

    def save_experience(self, experience: ExperienceRecord) -> None:
        self._backend.save(experience.id, experience)

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        return self._backend.get(experience_id)

    def list_experiences(self, limit: int | None = None) -> list[ExperienceRecord]:
        experiences = self._backend.values()
        experiences.sort(key=lambda experience: experience.provenance.time or 0.0, reverse=True)
        if limit is not None:
            experiences = experiences[: max(0, limit)]
        return experiences

    def search(self, text: str, k: int = 5) -> list[ExperienceRecord]:
        if k <= 0:
            raise ValueError("k must be > 0")
        terms = _terms(text)
        if not terms:
            return self.list_experiences(limit=k)

        scored: list[tuple[int, float, ExperienceRecord]] = []
        for experience in self._backend.values():
            haystack = _experience_text(experience)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, experience.provenance.time or 0.0, experience))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [experience for _, _, experience in scored[:k]]

    def count(self) -> int:
        return self._backend.count()

    def clear(self) -> int:
        return self._backend.clear()


class PersistentExperienceRepo(ExperienceRepo):
    def __init__(self, inner: ExperienceRepo, path: str | Path) -> None:
        super().__init__(backend=JsonRecordStoreBackend(path, ExperienceRecord))
        for item in inner.list_experiences():
            self.save_experience(item)


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in str(text or "").casefold().split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 3:
            terms.append(term)
    return terms


def _experience_text(experience: ExperienceRecord) -> str:
    parts = [
        experience.goal,
        experience.context,
        experience.decision,
        *experience.actions,
        experience.outcome,
        json.dumps(experience.evaluation, ensure_ascii=False, sort_keys=True),
        experience.lesson,
        *experience.reuse_when,
        *experience.avoid_when,
    ]
    return " ".join(str(part or "") for part in parts).casefold()
