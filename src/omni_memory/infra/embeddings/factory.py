from __future__ import annotations

from omni_memory.domain.model_ports import IEmbedder
from omni_memory.embeddings import HashEmbedder, STEmbedder


def build_embedder(
    backend: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    device: str | None = None,
) -> IEmbedder:
    """
    Build an embedder from configuration.

    Supported backends:
    - hash: deterministic local fallback, good for tests and demos
    - st / sentence_transformers: SentenceTransformers model
    - auto: try SentenceTransformers, fall back to HashEmbedder
    """
    normalized = (backend or "auto").lower().replace("-", "_")

    if normalized == "hash":
        return HashEmbedder()

    if normalized in {"st", "sentence_transformers", "sentence_transformer"}:
        return STEmbedder(model_name=model_name, device=device)

    if normalized == "auto":
        try:
            return STEmbedder(model_name=model_name, device=device)
        except Exception:
            return HashEmbedder()

    raise ValueError(f"Unsupported embedding backend: {backend}")
