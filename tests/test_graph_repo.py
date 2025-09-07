# tests/test_graph_repo.py
from domain.models import Fact, Provenance
from infra.graph_repo import GraphRepo

def make_fact(fid: str, s: str, p: str, o: str) -> Fact:
    return Fact(
        id=fid,
        subject=s,
        predicate=p,
        object=o,
        provenance=Provenance(source="test"),
        meta={},
    )

def test_save_and_query_by_subject():
    repo = GraphRepo()
    f1 = make_fact("f1", "Alice", "at", "lighthouse")
    f2 = make_fact("f2", "Alice", "near", "sea")
    f3 = make_fact("f3", "Bob", "at", "bridge")
    for f in (f1, f2, f3):
        repo.save_fact(f)

    by_alice = repo.query(subject="Alice")
    ids = {f.id for f in by_alice}
    assert ids == {"f1", "f2"}

def test_query_by_predicate_and_object():
    repo = GraphRepo()
    repo.save_fact(make_fact("a1", "Alice", "at", "lighthouse"))
    repo.save_fact(make_fact("a2", "Alice", "near", "sea"))
    repo.save_fact(make_fact("b1", "Bob", "at", "lighthouse"))

    res = repo.query(predicate="at", object="lighthouse")
    ids = {f.id for f in res}
    assert ids == {"a1", "b1"}

def test_upsert_by_id_updates_edge():
    repo = GraphRepo()
    f = make_fact("x1", "Alice", "at", "lighthouse")
    repo.save_fact(f)
    # обновим predicate по тому же id (например, уточнение)
    f_updated = make_fact("x1", "Alice", "visited", "lighthouse")
    repo.save_fact(f_updated)

    out = repo.query(subject="Alice", object="lighthouse")
    assert len(out) == 1
    assert out[0].predicate == "visited"
