# tests/test_vector_repo.py
from domain.models import MemoryObject
from infra.vector_repo import VectorStoreRepo
from infra.exceptions import CapacityExceeded
import pytest 


def make_obj(obj_id: str, text: str) -> MemoryObject:
    return MemoryObject(id=obj_id, type="note", payload={"text": text})

def test_save_and_search_returns_relevant_items():
    repo = VectorStoreRepo()

    o1 = make_obj("1", "Alice was seen near the lighthouse by the sea")
    o2 = make_obj("2", "There is an old bridge over the quiet river")
    o3 = make_obj("3", "Nikolai is a fisherman in the village")

    repo.save_object(o1)
    repo.save_object(o2)
    repo.save_object(o3)

    # Запрос про маяк должен сильнее матчиться на o1
    hits = repo.semantic_search("Alice at the lighthouse", k=2)
    ids = [h.id for h in hits]

    assert "1" in ids
    # и точно что-то возвращаем
    assert len(hits) >= 1

def test_empty_index_returns_empty_list():
    repo = VectorStoreRepo()
    hits = repo.semantic_search("anything", k=5)
    assert hits == []

def test_capacity_limit_raises():
    repo = VectorStoreRepo(max_elements=1)
    repo.save_object(make_obj("1", "first"))
    with pytest.raises(CapacityExceeded):
        repo.save_object(make_obj("2", "second"))

