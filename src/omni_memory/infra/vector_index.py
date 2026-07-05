from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorIndexBackend(Protocol):
    """Small facade over concrete vector index libraries.

    Repositories should depend on this protocol instead of depending on FAISS,
    NumPy search, or another concrete vector database client directly.
    """

    @property
    def dim(self) -> int: ...

    @property
    def count(self) -> int: ...

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[int]: ...

    def reset(self) -> None: ...

    def save(self, dir_path: str) -> None: ...

    def load(self, dir_path: str) -> None: ...


def build_vector_index_backend(dim: int, prototype: VectorIndexBackend | None = None) -> VectorIndexBackend:
    """Build the default vector index backend or return an injected prototype."""
    if prototype is not None:
        if int(prototype.dim) != int(dim):
            raise ValueError(f"Vector index backend dim mismatch: backend={prototype.dim}, repo={dim}")
        return prototype
    if _faiss_available():
        return FaissVectorIndexBackend(dim)
    return NumpyVectorIndexBackend(dim)


class FaissVectorIndexBackend:
    def __init__(self, dim: int) -> None:
        import faiss  # type: ignore

        self._faiss = faiss
        self._dim = int(dim)
        self._index = self._faiss.IndexFlatIP(self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(_as_matrix(vectors, self._dim))

    def search(self, query: np.ndarray, k: int) -> list[int]:
        if self.count == 0:
            return []
        _distances, indices = self._index.search(_as_matrix(query, self._dim), min(k, self.count))
        return [int(idx) for idx in indices[0].tolist() if int(idx) >= 0]

    def reset(self) -> None:
        self._index = self._faiss.IndexFlatIP(self._dim)

    def save(self, dir_path: str) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "index.faiss"))
        _write_backend_meta(path, backend="faiss", dim=self._dim, count=self.count)

    def load(self, dir_path: str) -> None:
        path = Path(dir_path)
        self._index = self._faiss.read_index(str(path / "index.faiss"))


class NumpyVectorIndexBackend:
    def __init__(self, dim: int) -> None:
        self._dim = int(dim)
        self._vectors = np.empty((0, self._dim), dtype="float32")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        matrix = _as_matrix(vectors, self._dim)
        self._vectors = np.vstack([self._vectors, matrix])

    def search(self, query: np.ndarray, k: int) -> list[int]:
        if self.count == 0:
            return []
        q = _as_matrix(query, self._dim)
        scores = q @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, : min(k, self.count)]
        return [int(idx) for idx in order[0].tolist()]

    def reset(self) -> None:
        self._vectors = np.empty((0, self._dim), dtype="float32")

    def save(self, dir_path: str) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "index.npz", vectors=self._vectors)
        _write_backend_meta(path, backend="numpy", dim=self._dim, count=self.count)

    def load(self, dir_path: str) -> None:
        path = Path(dir_path)
        loaded = np.load(path / "index.npz")
        self._vectors = _as_matrix(loaded["vectors"], self._dim)


def _as_matrix(vectors: np.ndarray, dim: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise ValueError(f"Expected vectors with shape (*, {dim}), got {matrix.shape}")
    return matrix


def _write_backend_meta(path: Path, *, backend: str, dim: int, count: int) -> None:
    (path / "vector_backend.json").write_text(
        json.dumps({"backend": backend, "dim": int(dim), "count": int(count)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
