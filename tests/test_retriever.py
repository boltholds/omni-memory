# tests/test_retriever.py
from domain.models import MemoryObject, Fact, Episode, EpisodeEvent, Provenance
from infra.repo.vector_repo import VectorStoreRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.episodic_repo import EpisodicRepo
from app.retriever import Retriever


def _obj(i: str, text: str) -> MemoryObject:
    return MemoryObject(id=i, type="note", payload={"text": text}, provenance=Provenance(source="test"))

def _fact(fid: str, s: str, p: str, o: str) -> Fact:
    return Fact(id=fid, subject=s, predicate=p, object=o, provenance=Provenance(source="test"))

def _episode(eid: str, parts, summary, events) -> Episode:
    return Episode(id=eid, participants=parts, summary=summary, events=events, provenance=Provenance(source="test"))


def test_retrieve_bundle_contains_items_from_all_sources():
    # Vector
    vrepo = VectorStoreRepo()
    vrepo.save_object(_obj("n1", "Alice was near the lighthouse"))
    vrepo.save_object(_obj("n2", "An old stone bridge stands over the river"))

    # Graph
    grepo = GraphRepo()
    grepo.save_fact(_fact("f1", "alice", "at", "lighthouse"))
    grepo.save_fact(_fact("f2", "bob", "at", "bridge"))

    # Episodic
    erepo = EpisodicRepo()
    ep = _episode(
        "e1",
        ["Alice", "Nikolai"],
        "Evening near the lighthouse",
        [EpisodeEvent(t=1.0, event_type="seen", summary="Alice met fisherman Nikolai", refs={})],
    )
    erepo.save_episode(ep)

    r = Retriever(vrepo, grepo, erepo)
    bundle = r.retrieve("Alice at the lighthouse", k_sem=3, k_eps=3)

    # semantic
    assert len(bundle.semantic_chunks) >= 1
    assert any(o.id == "n1" for o in bundle.semantic_chunks)

    # facts (must include Alice::lighthouse)
    assert any(f.id == "f1" for f in bundle.facts)

    # episodes (should prefer one with lighthouse)
    assert any(e.id == "e1" for e in bundle.episodes)


def test_retrieve_handles_empty_sources_gracefully():
    vrepo = VectorStoreRepo()
    grepo = GraphRepo()
    erepo = EpisodicRepo()
    r = Retriever(vrepo, grepo, erepo)

    bundle = r.retrieve("random query", k_sem=5, k_eps=2)
    assert bundle.semantic_chunks == []
    assert bundle.facts == []
    assert bundle.episodes == []
