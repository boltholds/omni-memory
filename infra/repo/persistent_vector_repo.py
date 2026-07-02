from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.models import MemoryObject
from infra.repo.vector_repo import VectorStoreRepo


class PersistentVectorRepo:
    """File-backed wrapper around VectorStoreRepo.

    VectorStoreRepo отвечает за векторный индекс и semantic search.
    PersistentVectorRepo отвечает за загрузку/сохранение snapshot между CLI-запусками.
    """

    def __init__(self, inner: VectorStoreRepo, dir_path: str | Path) -> None:
        self.inner = inner
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    def save_object(self, obj: MemoryObject) -> bool:
        saved = self.inner.save_object(obj)
        self._flush()
        return saved

    def semantic_search(self, text: str, k: int = 5) -> list[MemoryObject]:
        return self.inner.semantic_search(text, k=k)

    def is_duplicate_text(self, text: str) -> bool:
        return self.inner.is_duplicate_text(text)

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

    def save(self, dir_path: str | Path | None = None) -> None:
        self.inner.save(str(dir_path or self.dir_path))

    def load(self, dir_path: str | Path | None = None) -> None:
        self.inner.load(str(dir_path or self.dir_path))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _load_if_exists(self) -> None:
        required = [
            self.dir_path / "index.faiss",
            self.dir_path / "ids.json",
            self.dir_path / "store.json",
            self.dir_path / "meta.json",
        ]

        if not all(path.exists() for path in required):
            return

        self.inner.load(str(self.dir_path))

    def _flush(self) -> None:
        self.inner.save(str(self.dir_path))
