from __future__ import annotations

import time

from app.builder import build_memory
from domain.models import Episode, EpisodeEvent, Fact, Provenance
from infra.embeddings.factory import HashEmbedder
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo


def _memory_with_vector(vrepo: VectorStoreRepo):
    return build_memory(use_llm=False, embedder=HashEmbedder(), vector_repo=vrepo)


def test_note_dedup_and_gc_vector():
    vrepo = VectorStoreRepo(embedder=HashEmbedder())
    memory = _memory_with_vector(vrepo)

    items = [
        {"id": "n1", "type": "note", "text": "Stone bridge over river", "meta": {"volatility": "normal"}},
        {"id": "n2", "type": "note", "text": "stone   bridge over   river"},  # дубль
    ]
    result = memory.write_items_raw(items, source="codex-dev")

    assert result.saved_count == 1
    assert result.rejected_count == 1
    assert any("duplicate" in reason for reason in result.reasons)

    # истечём TTL у n1
    # жёстко выставим expire_at в прошлом
    obj = vrepo._store["n1"]
    obj.meta["expire_at"] = time.time() - 1

    removed = vrepo.gc_expired()
    assert removed == 1
    assert len(vrepo._store) == 0


def test_gc_graph_and_episodic():
    grepo = GraphRepo()
    erepo = EpisodicRepo()
    # вручную положим устаревшие записи
    fact = Fact(
        id="f1",
        subject="a",
        predicate="p",
        object="o",
        provenance=Provenance(),
        meta={"expire_at": time.time() - 10},
    )
    grepo.save_fact(fact)
    episode = Episode(
        id="e1",
        participants=["A"],
        summary="old",
        events=[EpisodeEvent(t=1.0, event_type="x", summary="y")],
        provenance=Provenance(),
        meta={"expire_at": time.time() - 10},
    )
    erepo.save_episode(episode)

    assert grepo.gc_expired() == 1
    assert erepo.gc_expired() == 1
