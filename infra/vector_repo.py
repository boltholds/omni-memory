# infra/vector_repo.py
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

import faiss  # type: ignore

from domain.models import MemoryObject
from domain.ports import IMemoryReadRepository, IMemoryWriteRepository
from app.embeddings import Embedder, HashEmbedder


def _text_from_payload(payload: dict) -> str:
    return (
        payload.get("text")
        or payload.get("content")
        or payload.get("body")
        or str(payload)
    )


class VectorStoreRepo(IMemoryReadRepository, IMemoryWriteRepository):
    """
    Хранит объекты в памяти, эмбеддинги — через переданный Embedder.
    По умолчанию HashEmbedder; для cosine используем IndexFlatIP.
    """
    def __init__(self, embedder: Optional[Embedder] = None, max_elements: int = 100_000) -> None:
        self._embedder: Embedder = embedder or HashEmbedder()
        self._dim = int(self._embedder.dim)
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids: List[str] = []
        self._store: Dict[str, MemoryObject] = {}
        self._max_elements = max_elements

    # ---- write ----
    def save_object(self, obj: MemoryObject) -> None:
        if len(self._ids) >= self._max_elements:
            raise RuntimeError("VectorStoreRepo capacity reached")
        payload = obj.payload or {}
        text = _text_from_payload(payload)
        emb = self._embedder.embed_one(str(text))[np.newaxis, :].astype("float32")
        self._index.add(emb)
        self._ids.append(obj.id)
        self._store[obj.id] = obj

    # ---- read ----
    def semantic_search(self, text: str, k: int = 5) -> List[MemoryObject]:
        if len(self._ids) == 0:
            return []
        q = self._embedder.embed_one(text)[np.newaxis, :].astype("float32")
        _D, I = self._index.search(q, min(k, len(self._ids)))
        out: List[MemoryObject] = []
        for row in I[0].tolist():
            if row == -1:
                continue
            obj_id = self._ids[row]
            obj = self._store.get(obj_id)
            if obj:
                out.append(obj)
        return out
