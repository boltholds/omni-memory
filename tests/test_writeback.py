from __future__ import annotations

from app.builder import build_memory
from domain.models import Provenance
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_writeback_accepts_verified_fact_and_rejects_pii_note():
    memory = _memory()

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

    result = memory.write_items_raw(items, source="test")

    assert result.saved_count == 1
    assert result.rejected_count == 1
    assert any("pii_email_blocked" in reason for reason in result.reasons)

    saved = result.saved[0]
    assert saved.meta["scope"]["environment"] == "test"
    assert saved.meta["scope"]["durability"] == "ephemeral"


def test_writeback_saves_episode_and_note_without_pii():
    memory = _memory()

    items = [
        {
            "id": "ep1",
            "participants": ["Alice", "Nikolai"],
            "summary": "Evening near the lighthouse",
            "events": [
                {"t": 1.0, "event_type": "seen", "summary": "Alice met fisherman Nikolai", "refs": {}}
            ],
            "provenance": {"source": "codex-dev"},
        },
        {"id": "n1", "type": "note", "text": "Stone bridge over quiet river"},
    ]

    result = memory.write_items_raw(items, source="codex-dev")

    assert result.saved_count == 2
    assert result.rejected_count == 0
    assert all(item.meta["scope"]["durability"] == "durable" for item in result.saved)


def test_writeback_result_contains_operations_and_policy_decisions():
    memory = _memory()

    result = memory.write_items_raw(
        [
            {
                "id": "fact-auditable",
                "type": "fact",
                "subject": "writeback",
                "predicate": "uses",
                "object": "policy pipeline",
                "meta": {"confidence": 1.0, "domain_ids": ["domain:project:omni-memory"]},
            }
        ],
        source="codex-dev",
    )

    assert result.saved_count == 1
    assert result.operations
    assert result.policy_decisions
    assert any(decision.stage == "conversion" for decision in result.policy_decisions)
    assert any(decision.stage == "repository" for decision in result.policy_decisions)
