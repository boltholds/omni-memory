# app/embeddings.py
from __future__ import annotations
from typing import List, Protocol, Optional
import numpy as np

class Embedder(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray: ...
    def embed_one(self, text: str) -> np.ndarray: ...

# ---------- Hash (лёгкий дефолт) ----------

class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            h = hash(tok)
            i = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            v[i] += sign
        # l2 normalize
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype("float32")

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._hash_vec(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._hash_vec(text)

# ---------- Sentence-Transformers (опционально) ----------

class STEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("sentence-transformers недоступен. Установи пакет или используй HashEmbedder.") from e
        self._model = SentenceTransformer(model_name, device=device)  # загружается лениво из HF
        # узнаем dim
        test = self._model.encode(["dim probe"], normalize_embeddings=True)
        self.dim = int(test.shape[1])

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

# ---------- Фабрика по настройкам ----------

def build_embedder(backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """
    backend: "auto" | "st" | "hash"
    """
    if backend == "hash":
        return HashEmbedder()
    if backend == "st":
        return STEmbedder(model_name=model_name)
    # auto: сначала пробуем ST, иначе hash
    try:
        return STEmbedder(model_name=model_name)
    except Exception:
        return HashEmbedder()
