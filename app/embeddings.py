# app/embeddings.py
from __future__ import annotations
import hashlib
import math
import re
from typing import List, Optional, Iterable
import numpy as np

from domain.model_ports import IEmbedder as Embedder

# ---------- Hash ----------

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        tokens = self._tokens(text)

        for token in tokens:
            yield f"u:{token}", 1.0

        for a, b in zip(tokens, tokens[1:]):
            yield f"b:{a}_{b}", 1.5

        for token in tokens:
            if len(token) < 3:
                continue

            wrapped = f"<{token}>"
            for i in range(len(wrapped) - 2):
                yield f"c3:{wrapped[i:i + 3]}", 0.25

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)

        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            n = int.from_bytes(digest, "big")

            idx = n % self.dim
            sign = 1.0 if ((n >> 63) & 1) == 0 else -1.0

            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec

        return (vec / norm).astype(np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        return np.vstack([self.embed_one(text) for text in texts]).astype(np.float32)

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

# ---------- Backward-compatible factory import ----------

def build_embedder(backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Compatibility wrapper. Prefer infra.embeddings.build_embedder."""
    from infra.embeddings.factory import build_embedder as _build_embedder

    return _build_embedder(backend=backend, model_name=model_name)
