# infra/vector_repo.py
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path
try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight test/dev fallback
    class _IndexFlatIP:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self._vectors = np.empty((0, dim), dtype="float32")

        def add(self, vectors) -> None:
            arr = np.asarray(vectors, dtype="float32")
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise ValueError(f"Expected vectors with shape (*, {self.dim})")
            self._vectors = np.vstack([self._vectors, arr])

        def search(self, query, k: int):
            q = np.asarray(query, dtype="float32")
            if self._vectors.shape[0] == 0:
                distances = np.empty((q.shape[0], 0), dtype="float32")
                indices = np.empty((q.shape[0], 0), dtype="int64")
                return distances, indices
            scores = q @ self._vectors.T
            order = np.argsort(-scores, axis=1)[:, :k]
            distances = np.take_along_axis(scores, order, axis=1).astype("float32")
            return distances, order.astype("int64")

    class _FaissFallback:
        IndexFlatIP = _IndexFlatIP

        @staticmethod
        def write_index(index, path: str) -> None:
            raise RuntimeError("faiss is not installed; vector index persistence is unavailable")

        @staticmethod
        def read_index(path: str):
            raise RuntimeError("faiss is not installed; vector index persistence is unavailable")

    faiss = _FaissFallback()  # type: ignore
import hashlib
import re
import time

from omni_memory.domain.models import MemoryObject
from omni_memory.domain.ports import IMemoryReadRepository, IMemoryWriteRepository
from omni_memory.embeddings import Embedder, HashEmbedder
from omni_memory.profiling import timed
from omni_memory.stats import stats
from omni_memory.metrics import VECTOR_SIZE
from omni_memory.infra.exceptions import CapacityExceeded, EmbedderDimMismatch, SnapshotCorrupted, PersistenceError

def _text_from_payload(payload: dict) -> str:
    return (
        payload.get("text")
        or payload.get("content")
        or payload.get("body")
        or str(payload)
    )


_norm_ws_re = re.compile(r"\s+")

def _text_signature(text: str) -> str:
    # нормализация: нижний регистр, удаление лишних пробелов, ограничение длины
    t = _norm_ws_re.sub(" ", text.strip().lower())
    t = t[:4096]
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


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
        self._sigs: Dict[str, str] = {}  # id -> signature
        self._sig_index: Dict[str, str] = {}  # signature -> id (первый встретившийся)
        self._max_elements = max_elements


    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> int:
        removed = self.count()
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids.clear()
        self._store.clear()
        self._sigs.clear()
        self._sig_index.clear()
        try:
            VECTOR_SIZE.set(self.count())
        except Exception:
            pass
        return removed
    
    # ---- write ----
    def save_object(self, obj: MemoryObject) -> bool:
        if len(self._ids) >= self._max_elements:
            raise CapacityExceeded(f"VectorStoreRepo capacity reached: {self._max_elements}")
        payload = obj.payload or {}
        text = _text_from_payload(payload)
        emb = self._embedder.embed_one(str(text))[np.newaxis, :].astype("float32")
        
        sig = _text_signature(str(text))
        # если уже существует такой текст — ничего не добавляем (дедуп)
        if sig in self._sig_index:
            # но обновим meta/объект (например, TTL) по id-дубликату
            dup_id = self._sig_index[sig]
            self._store[dup_id].meta = obj.meta
            return False
        # иначе сохраняем как новый
        self._sig_index[sig] = obj.id
        self._sigs[obj.id] = sig
                
        self._index.add(emb)
        self._ids.append(obj.id)
        self._store[obj.id] = obj
        try:
            VECTOR_SIZE.set(self.count())
        except Exception:
            pass
        finally:
            return True

    # ---- read ----
    @timed("retriever.retrieve", slow_ms=100)
    def semantic_search(self, text: str, k: int = 5) -> List[MemoryObject]:
        stop = stats.timeit("vector.search_ms")
        if k <= 0:
            raise ValueError("k must be > 0")
        try:
            if (sizeids := len(self._ids)) == 0:
                return []
            q = self._embedder.embed_one(text)[np.newaxis, :].astype("float32")
            _D, I = self._index.search(q, min(k, sizeids))
            out: List[MemoryObject] = []
            for row in I[0].tolist():
                if row == -1:
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
        """
        Сохранить индекс и метаданные в папку:
            - index.faiss  — бинарник индекса
            - ids.json     — список obj_id в порядке FAISS
            - store.json   — объекты MemoryObject (model_dump)
            - meta.json    — {'dim': ..., 'count': ...}
        """
        p = Path(dir_path)
        try:
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

        except (OSError, json.JSONDecodeError, TypeError) as e:
            raise PersistenceError(f"Failed to save snapshot to {p}") from e

    def load(self, dir_path: str) -> None:
        """
        Загрузить индекс и метаданные из папки. Требует совместимого embedder.dim.
        """
        from omni_memory.domain.models import MemoryObject  # локальный импорт, чтобы избежать циклов
        p = Path(dir_path)
        idx_path = p / "index.faiss"
        ids_path = p / "ids.json"
        store_path = p / "store.json"
        meta_path = p / "meta.json"

        if not (idx_path.exists() and ids_path.exists() and store_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Vector snapshot incomplete in {p}")

        # meta для проверки
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SnapshotCorrupted(f"Bad meta.json in {p}") from e
        
        
        dim = int(meta.get("dim", 0))
        if dim and dim != self._dim:
            raise EmbedderDimMismatch(f"Embedder dim mismatch: file={dim}, repo={self._dim}")

        # 1) индекс
        self._index = faiss.read_index(str(idx_path))
        # 2) ids
        self._ids = json.loads(ids_path.read_text(encoding="utf-8"))
        # 3) store
        raw_store = json.loads(store_path.read_text(encoding="utf-8")) or {}
        self._store = {k: MemoryObject.model_validate(v) for k, v in raw_store.items()}
        # capacity оставляем прежним

    def gc_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        to_remove: List[int] = []
        removed = 0
        # определим какие ids удалять
        dead_ids: List[str] = []
        for obj_id, obj in list(self._store.items()):
            exp = (obj.meta or {}).get("expire_at")
            if exp is not None and float(exp) < now:
                dead_ids.append(obj_id)
        if not dead_ids:
            return 0
        # удалить из faiss (нужно перестроить компактно: пересоберём индекс)
        # 1) отфильтруем живых
        keep_mask = [i for i, oid in enumerate(self._ids) if oid not in dead_ids]
        new_index = faiss.IndexFlatIP(self._dim)
        # 2) переиндексируем: заново добавим эмбеддинги живых объектов
        #    (для MVP извлекаем текст и встраиваем снова)
        self._index = new_index
        new_ids: List[str] = []
        for i in keep_mask:
            oid = self._ids[i]
            obj = self._store[oid]
            payload = obj.payload or {}
            text = (
                payload.get("text")
                or payload.get("content")
                or payload.get("body")
                or str(payload)
            )
            emb = self._embedder.embed_one(str(text))[np.newaxis, :].astype("float32")
            self._index.add(emb)
            new_ids.append(oid)
        self._ids = new_ids
        # 3) подчистим словари/сигнатуры
        for oid in dead_ids:
            removed += 1
            sig = self._sigs.pop(oid, None)
            if sig and self._sig_index.get(sig) == oid:
                self._sig_index.pop(sig, None)
            self._store.pop(oid, None)
            
        try:
            VECTOR_SIZE.set(self.count())
        except Exception:
            pass
        finally:
            return removed
