from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from omni_memory import OmniMemory
from omni_memory.domain.llm import LLMResult, Msg
from omni_memory.domain.model_ports import ModelBundle


class DummyLLM:
    def generate(self, messages: list[Msg], temperature: float = 0.3) -> LLMResult:
        return {"text": "dummy answer", "model": "dummy-llm", "finish_reason": "stop"}


class DistillingLLM:
    def generate(self, messages: list[Msg], temperature: float = 0.3) -> LLMResult:
        return {
            "text": """
            {
              "candidates": [
                {
                  "kind": "fact",
                  "should_write": true,
                  "confidence": 0.93,
                  "reason": "The project framework is explicitly stated.",
                  "evidence_quote": "OmniMemory uses FastAPI.",
                  "temporal_scope": "current",
                  "payload": {
                    "subject": "OmniMemory",
                    "predicate": "uses",
                    "object": "FastAPI"
                  }
                }
              ],
              "rejected": []
            }
            """,
            "model": "local-distiller",
            "finish_reason": "stop",
        }


class TinyEmbedder:
    dim = 8

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = 1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


class PreferredReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    def rerank(self, query: str, documents: Sequence[Any]) -> list[Any]:
        docs = list(documents)
        self.calls.append((query, docs))
        return sorted(docs, key=lambda item: "preferred" in _item_text(item).casefold(), reverse=True)


class CountingReranker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def rerank(self, query: str, documents: Sequence[Any]) -> list[Any]:
        docs = list(documents)
        self.batch_sizes.append(len(docs))
        return docs


class BrokenReranker:
    def rerank(self, query: str, documents: Sequence[Any]) -> list[Any]:
        raise RuntimeError("reranker unavailable")


def _item_text(item: Any) -> str:
    payload = getattr(item, "payload", {}) or {}
    if isinstance(payload, dict):
        return " ".join(str(value) for value in payload.values())
    return str(item)


def test_byom_bundle_can_supply_llm_and_embedder():
    memory = OmniMemory(
        model_bundle=ModelBundle(
            llm=DummyLLM(),
            embedder=TinyEmbedder(),
        )
    )

    assert memory.llm is not None
    assert memory.vector_repo._dim == 8

    answer = memory.ask("Where is Alice?")
    assert answer.answer == "dummy answer"
    assert answer.model == "dummy-llm"


def test_llm_bundle_can_power_session_distillation_without_explicit_distiller(monkeypatch):
    from omni_memory.config import settings

    monkeypatch.setattr(settings, "distiller_provider", "inherit")

    memory = OmniMemory(
        model_bundle=ModelBundle(
            llm=DistillingLLM(),
            embedder=TinyEmbedder(),
        )
    )

    memory.ingest_turn("user", "OmniMemory uses FastAPI.")
    result = memory.commit_session(source="test-distiller")

    assert result.saved_count == 1
    assert result.saved[0].subject == "omnimemory"
    assert result.saved[0].predicate == "uses"
    assert result.saved[0].object == "FastAPI"


def test_model_bundle_reranker_reorders_retrieved_semantic_chunks():
    reranker = PreferredReranker()
    memory = OmniMemory(model_bundle=ModelBundle(embedder=TinyEmbedder(), reranker=reranker))
    memory.write_note("ordinary dependency note", source="codex-dev")
    memory.write_note("preferred dependency note", source="codex-dev")

    result = memory.retrieve("dependency", k_sem=2, scope={"memory_types": ["note"]})

    assert reranker.calls
    assert result.semantic_chunks[0].payload["text"] == "preferred dependency note"
    assert result.semantic_chunks[1].payload["text"] == "ordinary dependency note"


def test_model_bundle_reranker_failure_falls_back_to_pre_ranked_order():
    memory = OmniMemory(model_bundle=ModelBundle(embedder=TinyEmbedder(), reranker=BrokenReranker()))
    memory.write_note("first dependency note", source="codex-dev")
    memory.write_note("second dependency note", source="codex-dev")

    result = memory.retrieve("dependency", k_sem=2, scope={"memory_types": ["note"]})

    assert len(result.semantic_chunks) == 2


def test_model_bundle_reranker_respects_fast_candidate_budget(monkeypatch):
    from omni_memory.config import settings

    monkeypatch.setattr(settings, "reranker_max_candidates_fast", 4)
    reranker = CountingReranker()
    memory = OmniMemory(model_bundle=ModelBundle(embedder=TinyEmbedder(), reranker=reranker))
    for idx in range(20):
        memory.write_note(f"dependency note {idx}", source="codex-dev")

    memory.retrieve("dependency", k_sem=10, mode="fast", scope={"memory_types": ["note"]})

    assert reranker.batch_sizes
    assert max(reranker.batch_sizes) == 4
