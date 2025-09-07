from pathlib import Path
from infra.vector_repo import VectorStoreRepo
from domain.models import MemoryObject, Provenance

def _obj(i, t): 
    return MemoryObject(id=str(i), type="note", payload={"text": t}, provenance=Provenance(source="test"))

def test_vector_save_load_roundtrip(tmp_path: Path):
    repo = VectorStoreRepo()  # HashEmbedder по умолчанию
    repo.save_object(_obj("1", "alice near lighthouse"))
    repo.save_object(_obj("2", "stone bridge over river"))

    snap = tmp_path / "vdb"
    repo.save(str(snap))

    # новый инстанс с тем же embedder dim
    repo2 = VectorStoreRepo()
    repo2.load(str(snap))

    hits = repo2.semantic_search("Where is the lighthouse?", k=2)
    ids = [h.id for h in hits]
    assert "1" in ids
