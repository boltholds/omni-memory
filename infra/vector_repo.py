# infra/vector_repo.py
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path
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


    def save(self, dir_path: str) -> None:
        """
        Сохранить индекс и метаданные в папку:
            - index.faiss  — бинарник индекса
            - ids.json     — список obj_id в порядке FAISS
            - store.json   — объекты MemoryObject (model_dump)
            - meta.json    — {'dim': ..., 'count': ...}
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        # 1) индекс
        faiss.write_index(self._index, str(p / "index.faiss"))
        # 2) ids
        (p / "ids.json").write_text(json.dumps(self._ids, ensure_ascii=False, indent=2), encoding="utf-8")
        # 3) store
        store_dump = {k: v.model_dump() for k, v in self._store.items()}
        (p / "store.json").write_text(json.dumps(store_dump, ensure_ascii=False), encoding="utf-8")
        # 4) meta
        meta = {"dim": self._dim, "count": len(self._ids)}
        (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, dir_path: str) -> None:
        """
        Загрузить индекс и метаданные из папки. Требует совместимого embedder.dim.
        """
        from domain.models import MemoryObject  # локальный импорт, чтобы избежать циклов
        p = Path(dir_path)
        idx_path = p / "index.faiss"
        ids_path = p / "ids.json"
        store_path = p / "store.json"
        meta_path = p / "meta.json"

        if not (idx_path.exists() and ids_path.exists() and store_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Vector snapshot incomplete in {p}")

        # meta для проверки
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", 0))
        if dim and dim != self._dim:
            raise RuntimeError(f"Embedder dim mismatch: file={dim}, repo={self._dim}")

        # 1) индекс
        self._index = faiss.read_index(str(idx_path))
        # 2) ids
        self._ids = json.loads(ids_path.read_text(encoding="utf-8"))
        # 3) store
        raw_store = json.loads(store_path.read_text(encoding="utf-8")) or {}
        self._store = {k: MemoryObject.model_validate(v) for k, v in raw_store.items()}
        # capacity оставляем прежним
