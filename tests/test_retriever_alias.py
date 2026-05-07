from app.retriever import Retriever
from infra.repo.vector_repo import VectorStoreRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.episodic_repo import EpisodicRepo
from domain.models import MemoryObject, Fact, Episode, EpisodeEvent, Provenance
from app.config import settings

def _obj(i, t): return MemoryObject(id=i, type="note", payload={"text":t}, provenance=Provenance(source="test"))
def _fact(i,s,p,o): return Fact(id=i, subject=s, predicate=p, object=o, provenance=Provenance(source="test"))
def _episode(eid, parts, summ, evs): return Episode(id=eid, participants=parts, summary=summ, events=evs, provenance=Provenance(source="test"))

def test_retriever_alias_beacon_matches_lighthouse(monkeypatch):
    # подменим алиасы в рантайме
    monkeypatch.setattr(settings, "entity_aliases", {"lighthouse":["beacon"]})
    monkeypatch.setattr(settings, "ner_backend", "regex")

    vrepo = VectorStoreRepo()
    grepo = GraphRepo()
    erepo = EpisodicRepo()

    grepo.save_fact(_fact("f1","alice","at","lighthouse"))
    erepo.save_episode(_episode("e1",["Alice"],"Evening near the lighthouse",[EpisodeEvent(t=1.0, event_type="seen", summary="ok")]))

    r = Retriever(vrepo, grepo, erepo)
    bundle = r.retrieve("Where is Alice? She was at the BEACON.", k_sem=3, k_eps=3)

    assert any(f.id == "f1" for f in bundle.facts)
    assert any(e.id == "e1" for e in bundle.episodes)
