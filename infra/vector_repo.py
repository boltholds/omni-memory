# infra/vector_repo.py
from __future__ import annotations

from typing import Dict, List
import numpy as np

import faiss  # type: ignore

from domain.models import MemoryObject
from domain.ports import IMemoryReadRepository, IMemoryWriteRepository


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    # v shape: (dim,) or (n, dim)
    if v.ndim == 1:
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / n).astype("float32")


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Лёгкая детерминированная 'хеш-эмбеддинга' без внешних зависимостей.
    Идея: hashing trick (bag-of-words -> фиксированный вектор размера dim).
    Достаточно для MVP и тестов; позже заменим на sentence-transformers.
    """
    vec = np.zeros(dim, dtype="float32")
    for tok in text.lower().split():
        # простой хеш → индекс
        h = hash(tok)
        idx = h % dim
        # signed bucket (+/-1), чтобы уменьшить коллизионный шум
        sign = 1.0 if (h >> 1) & 1 else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


class VectorStoreRepo(IMemoryReadRepository, IMemoryWriteRepository):
    """
    Хранит объекты в памяти, а эмбеддинги — в FAISS IndexFlatIP.
    Поиск: cosine (через IP после L2-нормализации).
    """

    def __init__(self, dim: int = 384, max_elements: int = 100_000) -> None:
        self._dim = dim
        # IndexFlatIP — простой и быстрый для MVP; позже можно IVF/HNSW
        self._index = faiss.IndexFlatIP(dim)
        self._ids: List[str] = []                 # map: faiss_row -> obj_id
        self._store: Dict[str, MemoryObject] = {} # id -> MemoryObject
        self._max_elements = max_elements

    # ---- IMemoryWriteRepository ----
    def save_object(self, obj: MemoryObject) -> None:
        if len(self._ids) >= self._max_elements:
            raise RuntimeError("VectorStoreRepo capacity reached")

        # извлекаем текст для эмбеддинга
        payload = obj.payload or {}
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("body")
            or str(payload)
        )

        emb = _hash_embed(str(text), dim=self._dim)[np.newaxis, :]  # shape (1, dim) float32
        self._index.add(emb)
        self._ids.append(obj.id)
        self._store[obj.id] = obj

    # ---- IMemoryReadRepository ----
    def semantic_search(self, text: str, k: int = 5) -> List[MemoryObject]:
        if len(self._ids) == 0:
            return []

        q = _hash_embed(text, dim=self._dim)[np.newaxis, :]  # (1, dim)
        # faiss возвращает (D, I): расстояния и индексы
        _dimensions, indexes = self._index.search(q, min(k, len(self._ids)))
        idxs = indexes[0].tolist()

        result: List[MemoryObject] = []
        for row in idxs:
            if row == -1:
                continue
            obj_id = self._ids[row]
            obj = self._store.get(obj_id)
            if obj:
                result.append(obj)
        return result
