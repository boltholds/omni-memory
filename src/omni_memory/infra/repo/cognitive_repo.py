from __future__ import annotations

import json
from pathlib import Path

from omni_memory.domain.models import FailurePatternRecord, SkillRecord


class SkillRepo:
    def __init__(self) -> None:
        self._store: dict[str, SkillRecord] = {}

    def save_skill(self, skill: SkillRecord) -> None:
        self._store[skill.id] = skill

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        return self._store.get(skill_id)

    def list_skills(self, limit: int | None = None) -> list[SkillRecord]:
        items = list(self._store.values())
        items.sort(key=lambda item: item.provenance.time or 0.0, reverse=True)
        return items[: max(0, limit)] if limit is not None else items

    def search(self, text: str, k: int = 5) -> list[SkillRecord]:
        terms = _terms(text)
        if not terms:
            return self.list_skills(limit=k)
        scored: list[tuple[int, float, SkillRecord]] = []
        for item in self._store.values():
            haystack = _skill_text(item)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, item.provenance.time or 0.0, item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in scored[:k]]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = len(self._store)
        self._store.clear()
        return removed


class FailurePatternRepo:
    def __init__(self) -> None:
        self._store: dict[str, FailurePatternRecord] = {}

    def save_failure_pattern(self, pattern: FailurePatternRecord) -> None:
        self._store[pattern.id] = pattern

    def get_failure_pattern(self, pattern_id: str) -> FailurePatternRecord | None:
        return self._store.get(pattern_id)

    def list_failure_patterns(self, limit: int | None = None) -> list[FailurePatternRecord]:
        items = list(self._store.values())
        items.sort(key=lambda item: item.provenance.time or 0.0, reverse=True)
        return items[: max(0, limit)] if limit is not None else items

    def search(self, text: str, k: int = 5) -> list[FailurePatternRecord]:
        terms = _terms(text)
        if not terms:
            return self.list_failure_patterns(limit=k)
        scored: list[tuple[int, float, FailurePatternRecord]] = []
        for item in self._store.values():
            haystack = _failure_pattern_text(item)
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, item.provenance.time or 0.0, item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in scored[:k]]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = len(self._store)
        self._store.clear()
        return removed


class PersistentSkillRepo:
    def __init__(self, inner: SkillRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def save_skill(self, skill: SkillRecord) -> None:
        self.inner.save_skill(skill)
        self._flush()

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        return self.inner.get_skill(skill_id)

    def list_skills(self, limit: int | None = None) -> list[SkillRecord]:
        return self.inner.list_skills(limit=limit)

    def search(self, text: str, k: int = 5) -> list[SkillRecord]:
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
            self.inner.save_skill(SkillRecord.model_validate(item))

    def _flush(self) -> None:
        data = [item.model_dump(mode="json") for item in self.inner.list_skills()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class PersistentFailurePatternRepo:
    def __init__(self, inner: FailurePatternRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def save_failure_pattern(self, pattern: FailurePatternRecord) -> None:
        self.inner.save_failure_pattern(pattern)
        self._flush()

    def get_failure_pattern(self, pattern_id: str) -> FailurePatternRecord | None:
        return self.inner.get_failure_pattern(pattern_id)

    def list_failure_patterns(self, limit: int | None = None) -> list[FailurePatternRecord]:
        return self.inner.list_failure_patterns(limit=limit)

    def search(self, text: str, k: int = 5) -> list[FailurePatternRecord]:
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
            self.inner.save_failure_pattern(FailurePatternRecord.model_validate(item))

    def _flush(self) -> None:
        data = [item.model_dump(mode="json") for item in self.inner.list_failure_patterns()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _terms(text: str) -> list[str]:
    out: list[str] = []
    for raw in str(text or "").casefold().split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 3:
            out.append(term)
    return out


def _skill_text(skill: SkillRecord) -> str:
    parts = [
        skill.name,
        skill.problem,
        *skill.procedure,
        *skill.reuse_when,
        *skill.avoid_when,
        *skill.evidence_ids,
    ]
    return " ".join(str(part or "") for part in parts).casefold()


def _failure_pattern_text(pattern: FailurePatternRecord) -> str:
    parts = [
        pattern.symptom,
        pattern.root_cause,
        pattern.fix,
        pattern.detection,
        *pattern.evidence_ids,
    ]
    return " ".join(str(part or "") for part in parts).casefold()
