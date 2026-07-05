from __future__ import annotations

import json
from pathlib import Path

from omni_memory.domain.models import ExperienceRecord


class ExperienceRepo:
    def __init__(self) -> None:
        self._store: dict[str, ExperienceRecord] = {}

    def save_experience(self, experience: ExperienceRecord) -> None:
        self._store[experience.id] = experience

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        return self._store.get(experience_id)

    def list_experiences(
        self,
        limit: int | None = None,
    ) -> list[ExperienceRecord]:
        experiences = list(self._store.values())
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
        for experience in self._store.values():
            haystack = _experience_text(experience)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, experience.provenance.time or 0.0, experience))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [experience for _, _, experience in scored[:k]]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = self.count()
        self._store.clear()
        return removed


class PersistentExperienceRepo:
    def __init__(self, inner: ExperienceRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def save_experience(self, experience: ExperienceRecord) -> None:
        self.inner.save_experience(experience)
        self._flush()

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        return self.inner.get_experience(experience_id)

    def list_experiences(self, limit: int | None = None) -> list[ExperienceRecord]:
        return self.inner.list_experiences(limit=limit)

    def search(self, text: str, k: int = 5) -> list[ExperienceRecord]:
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
            self.inner.save_experience(ExperienceRecord.model_validate(item))

    def _flush(self) -> None:
        data = [item.model_dump(mode="json") for item in self.inner.list_experiences()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
