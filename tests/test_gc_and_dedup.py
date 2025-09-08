import time
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from app.writeback import WriteBackService
from domain.policy import MemoryPolicy

def test_note_dedup_and_gc_vector():
    vrepo = VectorStoreRepo()
    wb = WriteBackService(vrepo, GraphRepo(), EpisodicRepo(), MemoryPolicy())

    items = [
        {"id":"n1","type":"note","text":"Stone bridge over river","meta":{"volatility":"normal"}},
        {"id":"n2","type":"note","text":"stone   bridge over   river"},  # дубль
    ]
    rep = wb.write(items)
    assert rep.saved == 1
    assert any("duplicate_note" in r for r in rep.reasons)

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
    from domain.models import Fact, Episode, EpisodeEvent, Provenance
    f = Fact(id="f1", subject="a", predicate="p", object="o", provenance=Provenance(), meta={"expire_at": time.time()-10})
    grepo.save_fact(f)
    e = Episode(id="e1", participants=["A"], summary="old", events=[EpisodeEvent(t=1.0, event_type="x", summary="y")], provenance=Provenance(), meta={"expire_at": time.time()-10})
    erepo.save_episode(e)

    assert grepo.gc_expired() == 1
    assert erepo.gc_expired() == 1
