from __future__ import annotations

import sys
import types
from typing import Any

from omni_memory.builder import build_memory
from omni_memory.config import settings
from omni_memory.domain.models import MemoryObject, Provenance
from omni_memory.infra.embeddings.factory import HashEmbedder
from omni_memory.infra.rerankers.factory import build_reranker, reranker_candidate_budget


def test_reranker_factory_default_none_does_not_import_cross_encoder_module():
    sys.modules.pop("omni_memory.infra.rerankers.cross_encoder", None)

    assert build_reranker(provider="none") is None
    assert "omni_memory.infra.rerankers.cross_encoder" not in sys.modules


def test_cross_encoder_reranker_lazy_adapter_orders_documents(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

        def predict(self, pairs):
            return [10.0 if "preferred" in text else 1.0 for _query, text in pairs]

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = build_reranker(
        provider="cross-encoder",
        model_name="fake-cross-encoder",
        device="cpu",
    )
    docs = [
        MemoryObject(id="ordinary", type="note", payload={"text": "ordinary note"}, provenance=Provenance(source="test")),
        MemoryObject(id="preferred", type="note", payload={"text": "preferred note"}, provenance=Provenance(source="test")),
    ]

    assert reranker is not None
    assert [item.id for item in reranker.rerank("query", docs)] == ["preferred", "ordinary"]


def test_build_memory_uses_configured_reranker_when_bundle_does_not_supply_one(monkeypatch):
    class FakeReranker:
        def rerank(self, query, documents):
            return list(documents)

    fake = FakeReranker()
    monkeypatch.setattr("omni_memory.builder.build_reranker", lambda: fake)

    memory = build_memory(embedder=HashEmbedder())

    assert memory.reranker is fake


def test_settings_expose_reranker_group():
    assert settings.reranker.provider == settings.reranker_provider
    assert settings.reranker.model == settings.reranker_model
    assert settings.reranker.max_candidates_fast == settings.reranker_max_candidates_fast


def test_reranker_candidate_budget_uses_mode_specific_settings(monkeypatch):
    monkeypatch.setattr(settings, "reranker_max_candidates_fast", 7)
    monkeypatch.setattr(settings, "reranker_max_candidates_quality", 31)
    monkeypatch.setattr(settings, "reranker_max_candidates_offline", 99)

    assert reranker_candidate_budget("fast") == 7
    assert reranker_candidate_budget("quality") == 31
    assert reranker_candidate_budget("offline") == 99
    assert reranker_candidate_budget(None) == 7
