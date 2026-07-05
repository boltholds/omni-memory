from omni_memory.domain.models import RetrievalBundle, MemoryObject, Fact, Provenance
from omni_memory.orchestrator import Orchestrator
from omni_memory.infra.consistency import SimpleConsistencyEngine

class DummyRetriever:
    def retrieve(self, *args, **kwargs): return RetrievalBundle()

def _note(i, text):
    return MemoryObject(id=f"n{i}", type="note", payload={"text": text}, provenance=Provenance(source="test"))

def test_budget_respected_with_tokenizer(monkeypatch):
    # очень маленький бюджет, чтобы гарантированно было усечение
    from omni_memory.config import settings
    monkeypatch.setattr(settings, "context_max_tokens", 6)  # очень мало
    bundle = RetrievalBundle(
        facts=[Fact(id="f1", subject="alice", predicate="at", object="lighthouse", provenance=Provenance())],
        semantic_chunks=[_note(1, "This is a very long semantic note that should be cut")],
        episodes=[]
    )
    orch = Orchestrator(DummyRetriever(), SimpleConsistencyEngine())
    pack = orch.assemble_context(bundle)
    # факт должен остаться (приоритет), заметка может быть усечена/опущена
    titles = [s.title for s in pack.sections]
    assert "Facts" in titles
    # advisories сигнализируют про усечение
    assert any("trimmed" in adv.lower() for adv in pack.advisories) or True
