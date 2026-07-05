from omni_memory.domain.models import RetrievalBundle, MemoryObject, Fact, Provenance
from omni_memory.orchestrator import Orchestrator
from omni_memory.infra.consistency import SimpleConsistencyEngine

class DummyRetriever:
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        return RetrievalBundle()

def _note(i, text):
    return MemoryObject(id=f"n{i}", type="note", payload={"text": text}, provenance=Provenance(source="test"))

def test_context_respects_priority_and_budget(monkeypatch):
    # малый бюджет
    from omni_memory.config import settings
    monkeypatch.setattr(settings, "context_max_tokens", 10)

    bundle = RetrievalBundle(
        semantic_chunks=[_note(1, "a b c d e f g h i j k")],
        facts=[
            Fact(id="f1", subject="alice", predicate="at", object="lighthouse", provenance=Provenance()),
            Fact(id="f2", subject="alice", predicate="at", object="bridge", provenance=Provenance()),
        ],
        episodes=[],
    )
    orch = Orchestrator(DummyRetriever(), SimpleConsistencyEngine())
    pack = orch.assemble_context(bundle)

    # Должны быть Conflicts (высший приоритет) и усечённые Facts, Notes может не остаться
    titles = [s.title for s in pack.sections]
    assert titles[0] == "Conflicts"
    assert "Facts" in titles
    # бюджет маленький -> advisories должны сообщить о тримминге
    assert any("trimmed" in adv.lower() for adv in pack.advisories)

def test_context_includes_notes_when_budget_allows(monkeypatch):
    from omni_memory.config import settings
    monkeypatch.setattr(settings, "context_max_tokens", 1000)
    bundle = RetrievalBundle(
        semantic_chunks=[_note(1, "alpha beta gamma"), _note(2, "delta epsilon")],
        facts=[],
        episodes=[],
    )
    orch = Orchestrator(DummyRetriever(), SimpleConsistencyEngine())
    pack = orch.assemble_context(bundle)
    titles = [s.title for s in pack.sections]
    assert "Semantic Notes" in titles
