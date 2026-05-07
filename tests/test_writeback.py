# tests/test_writeback.py
from domain.models import Provenance
from domain.policy import MemoryPolicy
from infra.repo.vector_repo import VectorStoreRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.episodic_repo import EpisodicRepo
from app.writeback_legacy import WriteBackService



def test_writeback_accepts_verified_fact_and_rejects_pii_note():
    wb = WriteBackService(
        vector_repo=VectorStoreRepo(),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(),
        policy=MemoryPolicy(),  # accept=0.6
    )

    items = [
        # факт с высоким доверием
        {
            "id": "f-ok",
            "subject": "alice",
            "predicate": "at",
            "object": "lighthouse",
            "provenance": Provenance(source="verified", time=100.0).model_dump(),
        },
        # заметка с email -> должна быть отклонена по PII
        {"id": "n-bad", "type": "note", "text": "contact me at root@example.com"},
    ]

    rep = wb.write(items)
    assert rep.saved == 1
    assert rep.rejected == 1
    assert any("pii_blocked_note" in r for r in rep.reasons)


def test_writeback_saves_episode_and_note_without_pii():
    wb = WriteBackService(
        vector_repo=VectorStoreRepo(),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(),
    )

    items = [
        {
            "id": "ep1",
            "participants": ["Alice", "Nikolai"],
            "summary": "Evening near the lighthouse",
            "events": [
                {"t": 1.0, "event_type": "seen", "summary": "Alice met fisherman Nikolai", "refs": {}}
            ],
            "provenance": {"source": "test"},
        },
        {"id": "n1", "type": "note", "text": "Stone bridge over quiet river"},
    ]

    rep = wb.write(items)
    assert rep.saved == 2
    assert rep.rejected == 0
