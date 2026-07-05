# infra/vector_repo.py
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from omni_memory.domain.models import MemoryObject
from omni_memory.domain.ports import IMemoryReadRepository, IMemoryWriteRepository
from omni_memory.embeddings import Embedder, HashEmbedder
from omni_memory.infra.exceptions import CapacityExceeded, EmbedderDimMismatch, PersistenceError, SnapshotCorrupted
from omni_memory.infra.vector_index import VectorIndexBackend, build_vector_index_backend
from omni_memory.metrics import VECTOR_SIZE
from omni_memory.profiling import timed
from omni_memory.stats import stats


def _text_from_payload(payload: dict) -> str:
    return (
        payload.get("text")
        or payload.get("content")
        or payload.get("body")
        or str(payload)
    )


_norm_ws_re = re.compile(r"\s+")


def _text_signature(text: str) -> str:
    t = _norm_ws_re.sub(" ", text.strip().lower())
    t = t[:4096]
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


class VectorStoreRepo(IMemoryReadRepository, IMemoryWriteRepository):
    """Semantic memory repository backed by an injectable vector index facade."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        max_elements: int = 100_000,
        index_backend: VectorIndexBackend | None = None,
    ) -> None:
        self._embedder: Embedder = embedder or HashEmbedder()
        self._dim = int(self._embedder.dim)
        self._index = build_vector_index_backend(self._dim, prototype=index_backend)
        self._ids: List[str] = []
        self._store: Dict[str, MemoryObject] = {}
        self._sigs: Dict[str, str] = {}
        self._sig_index: Dict[str, str] = {}
        self._max_elements = max_elements

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> int:
        removed = self.count()
        self._index.reset()
        self._ids.clear()
        self._store.clear()
        self._sigs.clear()
        self._sig_index.clear()
        self._set_metric()
        return removed

    def save_object(self, obj: MemoryObject) -> bool:
        if len(self._ids) >= self._max_elements:
            raise CapacityExceeded(f"VectorStoreRepo capacity reached: {self._max_elements}")

        payload = obj.payload or {}
        text = _text_from_payload(payload)
        sig = _text_signature(str(text))
        if sig in self._sig_index:
            dup_id = self._sig_index[sig]
            self._store[dup_id].meta = obj.meta
            return False

        emb = self._embedder.embed_one(str(text))[np.newaxis, :].astype("float32")
        self._index.add(emb)
        self._ids.append(obj.id)
        self._store[obj.id] = obj
        self._sig_index[sig] = obj.id
        self._sigs[obj.id] = sig
        self._set_metric()
        return True

    @timed("retriever.retrieve", slow_ms=100)
    def semantic_search(self, text: str, k: int = 5) -> List[MemoryObject]:
        stop = stats.timeit("vector.search_ms")
        if k <= 0:
            raise ValueError("k must be > 0")
        try:
            if len(self._ids) == 0:
                return []
            q = self._embedder.embed_one(text)[np.newaxis, :].astype("float32")
            out: List[MemoryObject] = []
            for row in self._index.search(q, min(k, len(self._ids))):
                if row < 0 or row >= len(self._ids):
                    continue
                obj_id = self._ids[row]
                obj = self._store.get(obj_id)
                if obj:
                    out.append(obj)
            return out
        finally:
            stop()

    def is_duplicate_text(self, text: str) -> bool:
        return _text_signature(text) in self._sig_index

    def save(self, dir_path: str) -> None:
        """Save vector index and MemoryObject metadata into a snapshot directory."""
        p = Path(dir_path)
        try:
            p.mkdir(parents=True, exist_ok=True)
            self._index.save(str(p))
            (p / "ids.json").write_text(json.dumps(self._ids, ensure_ascii=False, indent=2), encoding="utf-8")
            store_dump = {k: v.model_dump() for k, v in self._store.items()}
            (p / "store.json").write_text(json.dumps(store_dump, ensure_ascii=False), encoding="utf-8")
            meta = {"dim": self._dim, "count": len(self._ids)}
            (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise PersistenceError(f"Failed to save snapshot to {p}") from exc

    def load(self, dir_path: str) -> None:
        from omni_memory.domain.models import MemoryObject

        p = Path(dir_path)
        ids_path = p / "ids.json"
        store_path = p / "store.json"
        meta_path = p / "meta.json"
        if not (ids_path.exists() and store_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Vector snapshot incomplete in {p}")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotCorrupted(f"Bad meta.json in {p}") from exc

        dim = int(meta.get("dim", 0))
        if dim and dim != self._dim:
            raise EmbedderDimMismatch(f"Embedder dim mismatch: file={dim}, repo={self._dim}")

        self._index.load(str(p))
        self._ids = json.loads(ids_path.read_text(encoding="utf-8"))
        raw_store = json.loads(store_path.read_text(encoding="utf-8")) or {}
        self._store = {k: MemoryObject.model_validate(v) for k, v in raw_store.items()}
        self._rebuild_signatures()
        self._set_metric()

    def gc_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        dead_ids: List[str] = []
        for obj_id, obj in list(self._store.items()):
            exp = (obj.meta or {}).get("expire_at")
            if exp is not None and float(exp) < now:
                dead_ids.append(obj_id)
        if not dead_ids:
            return 0

        keep_mask = [i for i, oid in enumerate(self._ids) if oid not in dead_ids]
        self._index.reset()
        new_ids: List[str] = []
        for i in keep_mask:
            oid = self._ids[i]
            obj = self._store[oid]
            text = _text_from_payload(obj.payload or {})
            emb = self._embedder.embed_one(str(text))[np.newaxis, :].astype("float32")
            self._index.add(emb)
            new_ids.append(oid)
        self._ids = new_ids

        removed = 0
        for oid in dead_ids:
            removed += 1
            sig = self._sigs.pop(oid, None)
            if sig and self._sig_index.get(sig) == oid:
                self._sig_index.pop(sig, None)
            self._store.pop(oid, None)

        self._set_metric()
        return removed

    def _rebuild_signatures(self) -> None:
        self._sigs.clear()
        self._sig_index.clear()
        for obj_id in self._ids:
            obj = self._store.get(obj_id)
            if obj is None:
                continue
            sig = _text_signature(_text_from_payload(obj.payload or {}))
            self._sigs[obj_id] = sig
            self._sig_index.setdefault(sig, obj_id)

    def _set_metric(self) -> None:
        try:
            VECTOR_SIZE.set(self.count())
        except Exception:
            pass
