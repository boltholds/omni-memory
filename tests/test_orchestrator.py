# tests/test_orchestrator.py
from domain.models import MemoryObject, Fact, Episode, EpisodeEvent, Provenance, RetrievalBundle
from infra.consistency import SimpleConsistencyEngine
from app.orchestrator import Orchestrator


class DummyRetriever:
    def __init__(self, bundle: RetrievalBundle):
        self._bundle = bundle
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        return self._bundle


def test_assemble_context_with_all_sections():
    bundle = RetrievalBundle(
        semantic_chunks=[MemoryObject(id="n1", type="note", payload={"text": "Alice saw lighthouse"})],
        facts=[
            Fact(id="f1", subject="Alice", predicate="at", object="lighthouse", provenance=Provenance()),
            Fact(id="f2", subject="Alice", predicate="at", object="bridge", provenance=Provenance()),
        ],
        episodes=[
            Episode(
                id="e1",
                participants=["Alice"],
                summary="Evening near lighthouse",
                events=[EpisodeEvent(t=1.0, event_type="seen", summary="Alice met Nikolai")],
                provenance=Provenance(),
            )
        ],
    )
    orch = Orchestrator(DummyRetriever(bundle), SimpleConsistencyEngine())

    result = orch.assemble_context(bundle)

    titles = [s.title for s in result.sections]
    assert "Semantic Notes" in titles
    assert "Facts" in titles
    assert "Conflicts" in titles
    assert "Episodes" in titles


def test_plan_retrieval_delegates_to_retriever():
    dummy = RetrievalBundle()
    orch = Orchestrator(DummyRetriever(dummy), SimpleConsistencyEngine())

    out = orch.plan_retrieval("test")
    assert isinstance(out, RetrievalBundle)
