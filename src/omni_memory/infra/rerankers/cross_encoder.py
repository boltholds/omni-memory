from __future__ import annotations

from typing import Any, Sequence

from omni_memory.domain.model_ports import IReranker


class SentenceTransformersCrossEncoderReranker(IReranker):
    """Lazy sentence-transformers CrossEncoder reranker.

    The heavy dependency and model are loaded only when this adapter is
    constructed, never on the default no-reranker path.
    """

    def __init__(self, *, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - exact import error varies
            raise RuntimeError(
                "sentence-transformers CrossEncoder is unavailable. "
                "Install sentence-transformers or set RERANKER_PROVIDER=none."
            ) from exc

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        self._model = CrossEncoder(model_name, **kwargs)

    def rerank(self, query: str, documents: Sequence[Any]) -> list[Any]:
        docs = list(documents)
        if len(docs) <= 1:
            return docs
        pairs = [(query, _document_text(item)) for item in docs]
        scores = list(self._model.predict(pairs))
        indexed = list(enumerate(zip(scores, docs)))
        indexed.sort(key=lambda row: (float(row[1][0]), -row[0]), reverse=True)
        return [item for _idx, (_score, item) in indexed]


def _document_text(item: Any) -> str:
    if hasattr(item, "payload"):
        payload = getattr(item, "payload", {}) or {}
        if isinstance(payload, dict):
            return " ".join(str(value) for value in payload.values())
        return str(payload)
    if hasattr(item, "model_dump"):
        return str(item.model_dump(mode="json"))
    return str(item)
