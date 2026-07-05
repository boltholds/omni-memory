# tests/test_orchestrator.py
from typing import Any

from omni_memory.domain.models import MemoryObject, Fact, Episode, EpisodeEvent, Provenance, RetrievalBundle
from omni_memory.infra.consistency import SimpleConsistencyEngine
from omni_memory.orchestrator import Orchestrator


class DummyRetriever:
    def __init__(self, bundle: RetrievalBundle):
        self._bundle = bundle
        self.last_call: dict[str, Any] | None = None

    def retrieve(
        self,
        query: str,
        k_sem: int = 5,
        k_eps: int = 3,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        self.last_call = {
            "query": query,
            "k_sem": k_sem,
            "k_eps": k_eps,
            "intent": intent,
            "mode": mode,
            "scope": scope,
        }
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
    retriever = DummyRetriever(dummy)
    orch = Orchestrator(retriever, SimpleConsistencyEngine())

    out = orch.plan_retrieval(
        "test",
        intent="write_code",
        mode="debug",
        scope={"memory_types": ["skill"]},
    )

    assert isinstance(out, RetrievalBundle)
    assert retriever.last_call == {
        "query": "test",
        "k_sem": 5,
        "k_eps": 3,
        "intent": "write_code",
        "mode": "debug",
        "scope": {"memory_types": ["skill"]},
    }
