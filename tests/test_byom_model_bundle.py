from __future__ import annotations

import numpy as np

from app.memory import OmniMemory
from domain.llm import LLMResult, Msg
from domain.model_ports import ModelBundle


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
    from app.config import settings

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
