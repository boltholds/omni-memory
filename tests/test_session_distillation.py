from __future__ import annotations

from domain.distiller import MemoryCandidate, SessionDistillationResult, SessionTurn
from app.memory import OmniMemory
from infra.embeddings.factory import HashEmbedder
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo


class FakeSessionDistiller:
    def distill(self, text: str):  # compatibility with IMemoryDistiller-like users
        raise NotImplementedError

    def distill_session(self, turns: list[SessionTurn]) -> SessionDistillationResult:
        return SessionDistillationResult(
            candidates=[
                MemoryCandidate(
                    kind="fact",
                    should_write=True,
                    confidence=0.92,
                    reason="current project framework was explicitly stated",
                    evidence_quote="Мы переехали на FastAPI, Flask больше не используем.",
                    temporal_scope="current",
                    payload={
                        "subject": "project",
                        "predicate": "framework",
                        "object": "FastAPI",
                    },
                ),
                MemoryCandidate(
                    kind="fact",
                    should_write=True,
                    confidence=0.91,
                    reason="bad candidate: evidence is not quoted from transcript",
                    evidence_quote="project uses Django",
                    temporal_scope="current",
                    payload={
                        "subject": "project",
                        "predicate": "framework",
                        "object": "Django",
                    },
                ),
                MemoryCandidate(
                    kind="note",
                    should_write=True,
                    confidence=0.25,
                    reason="too uncertain",
                    evidence_quote="Мы переехали на FastAPI, Flask больше не используем.",
                    temporal_scope="unknown",
                    payload={"text": "Maybe the user likes FastAPI."},
                ),
            ]
        )


def _memory() -> OmniMemory:
    embedder = HashEmbedder()
    return OmniMemory(
        use_llm=False,
        distiller=FakeSessionDistiller(),
        vector_repo=VectorStoreRepo(embedder=embedder),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(db_path=":memory:"),
    )


def test_commit_session_writes_only_valid_high_confidence_candidates():
    mem = _memory()
    mem.ingest_turn("user", "Проект сначала был на Flask.")
    mem.ingest_turn("user", "Мы переехали на FastAPI, Flask больше не используем.")

    result = mem.commit_session(source="eval:test")

    assert result.saved_count == 1
    assert result.rejected_count == 2
    assert result.saved[0].subject == "project"
    assert result.saved[0].predicate == "framework"
    assert result.saved[0].object == "FastAPI"
    assert any("evidence_quote_not_found" in reason for reason in result.reasons)
    assert any("candidate_confidence_too_low" in reason for reason in result.reasons)
