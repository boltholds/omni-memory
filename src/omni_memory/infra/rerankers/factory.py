from __future__ import annotations

from omni_memory.config import settings
from omni_memory.domain.model_ports import IReranker


def reranker_candidate_budget(mode: str | None = None) -> int:
    normalized = (mode or "fast").strip().casefold().replace("_", "-")
    if normalized in {"quality", "deep"}:
        return int(settings.reranker_max_candidates_quality)
    if normalized in {"offline", "benchmark"}:
        return int(settings.reranker_max_candidates_offline)
    return int(settings.reranker_max_candidates_fast)


def build_reranker(
    provider: str | None = None,
    model_name: str | None = None,
    *,
    device: str | None = None,
) -> IReranker | None:
    normalized = (provider if provider is not None else settings.reranker_provider)
    normalized = (normalized or "none").strip().casefold().replace("_", "-")
    model = model_name or settings.reranker_model
    selected_device = device if device is not None else settings.reranker_device

    if normalized in {"", "none", "off", "disabled"}:
        return None

    if normalized in {"sentence-transformers", "sentence-transformer", "st", "cross-encoder", "crossencoder"}:
        from omni_memory.infra.rerankers.cross_encoder import SentenceTransformersCrossEncoderReranker

        return SentenceTransformersCrossEncoderReranker(
            model_name=model,
            device=selected_device,
        )

    raise ValueError(f"Unsupported reranker provider: {provider}")
