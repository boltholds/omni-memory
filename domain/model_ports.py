from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from domain.distiller import IMemoryDistiller
from domain.llm import ILLMProvider


class IEmbedder(Protocol):
    """Embedding model port used by vector storage and retrieval."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


class IReranker(Protocol):
    """Optional reranker port for BYOM retrieval pipelines."""

    def rerank(self, query: str, documents: Sequence[Any]) -> list[Any]: ...


@dataclass(slots=True)
class ModelBundle:
    """
    User-supplied model bundle.

    BYO-LLM is the simple case where only llm is provided.
    BYOM is the wider case where llm, embedder, reranker and distiller can all
    be supplied by the application instead of being built from environment settings.
    """

    llm: ILLMProvider | None = None
    embedder: IEmbedder | None = None
    reranker: IReranker | None = None
    distiller: IMemoryDistiller | None = None
