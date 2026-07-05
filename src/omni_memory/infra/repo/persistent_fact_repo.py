from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omni_memory.domain.models import Fact
from omni_memory.infra.repo.graph_repo import GraphRepo


class PersistentFactRepo:
    """File-backed wrapper around GraphRepo.
    Oтвечает только за сохранение/загрузку.
    """

    def __init__(self, inner: GraphRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load_into_inner()

    def save_fact(self, fact: Fact) -> None:
        self.inner.save_fact(fact)
        self._flush()

    def get_fact(self, fact_id: str) -> Fact | None:
        return self.inner.get_fact(fact_id)

    def remove_fact(self, fact_id: str) -> bool:
        removed = self.inner.remove_fact(fact_id)
        if removed:
            self._flush()
        return removed

    def query(self, **query_spec: Any) -> list[Fact]:
        return self.inner.query(**query_spec)

    def count(self) -> int:
        return self.inner.count()

    def clear(self) -> int:
        removed = self.inner.clear()
        self._flush()
        return removed

    def gc_expired(self, now: float | None = None) -> int:
        removed = self.inner.gc_expired(now)
        if removed:
            self._flush()
        return removed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _load_into_inner(self) -> None:
        if not self.path.exists():
            return

        raw_text = self.path.read_text(encoding="utf-8").strip()

        if not raw_text:
            return

        raw = json.loads(raw_text)

        if not isinstance(raw, list):
            raise ValueError(f"Expected facts list in {self.path}")

        for item in raw:
            fact = Fact.model_validate(item)
            self.inner.save_fact(fact)

    def _flush(self) -> None:
        facts = [
            fact.model_dump(mode="json")
            for fact in self.inner.query()
        ]

        self.path.write_text(
            json.dumps(facts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
