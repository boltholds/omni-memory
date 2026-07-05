from __future__ import annotations

from pathlib import Path

import numpy as np

from omni_memory import build_memory
from omni_memory.domain.models import MemoryObject
from omni_memory.embeddings import HashEmbedder
from omni_memory.infra.repo.vector_repo import VectorStoreRepo
from omni_memory.infra.vector_index import NumpyVectorIndexBackend


class RecordingVectorIndexBackend:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self.vectors: list[np.ndarray] = []
        self.search_calls: list[tuple[tuple[int, ...], int]] = []
        self.reset_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.append(np.asarray(vectors, dtype="float32"))

    def search(self, query: np.ndarray, k: int) -> list[int]:
        self.search_calls.append((tuple(query.shape), k))
        return list(range(min(k, self.count)))

    def reset(self) -> None:
        self.reset_calls += 1
        self.vectors.clear()

    def save(self, dir_path: str) -> None:
        Path(dir_path, "recording.index").write_text(str(self.count), encoding="utf-8")

    def load(self, dir_path: str) -> None:
        count = int(Path(dir_path, "recording.index").read_text(encoding="utf-8"))
        self.vectors = [np.zeros((1, self._dim), dtype="float32") for _ in range(count)]


def _obj(obj_id: str, text: str) -> MemoryObject:
    return MemoryObject(id=obj_id, type="note", payload={"text": text})


def test_vector_store_repo_uses_injected_index_backend():
    backend = RecordingVectorIndexBackend(dim=64)
    repo = VectorStoreRepo(embedder=HashEmbedder(dim=64), index_backend=backend)

    repo.save_object(_obj("one", "alice near lighthouse"))
    repo.save_object(_obj("two", "stone bridge"))

    hits = repo.semantic_search("lighthouse", k=1)

    assert len(backend.vectors) == 2
    assert backend.search_calls == [((1, 64), 1)]
    assert [hit.id for hit in hits] == ["one"]


def test_vector_store_repo_roundtrips_with_numpy_backend(tmp_path: Path):
    repo = VectorStoreRepo(embedder=HashEmbedder(dim=64), index_backend=NumpyVectorIndexBackend(dim=64))
    repo.save_object(_obj("one", "alice near lighthouse"))
    repo.save_object(_obj("two", "stone bridge"))

    snap = tmp_path / "vdb"
    repo.save(str(snap))

    restored = VectorStoreRepo(embedder=HashEmbedder(dim=64), index_backend=NumpyVectorIndexBackend(dim=64))
    restored.load(str(snap))

    hits = restored.semantic_search("lighthouse", k=2)
    assert "one" in [hit.id for hit in hits]


def test_build_memory_accepts_vector_index_backend_prototype():
    backend = RecordingVectorIndexBackend(dim=64)
    memory = build_memory(embedder=HashEmbedder(dim=64), vector_index_backend=backend)

    memory.write_note("agent memory with injectable vector backend", source="test")
    memory.retrieve("injectable backend", k_sem=1)

    assert backend.count == 1
    assert backend.search_calls == [((1, 64), 1)]
