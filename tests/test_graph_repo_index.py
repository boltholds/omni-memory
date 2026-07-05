from __future__ import annotations

from omni_memory.domain.models import Fact, Provenance
from omni_memory.infra.repo.graph_repo import GraphRepo


def fact(fid: str, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(id=fid, subject=subject, predicate=predicate, object=object_, provenance=Provenance(source="test"), meta={"confidence": 1.0})


def test_graph_repo_query_indexes_update_remove_and_gc():
    repo = GraphRepo()
    repo.save_fact(fact("f1", "omnimemory", "uses", "fastmcp"))
    repo.save_fact(fact("f2", "persona", "uses", "fastapi"))

    assert [item.id for item in repo.query(subject="omnimemory")] == ["f1"]
    assert [item.id for item in repo.query(object="fastapi")] == ["f2"]
    assert {item.id for item in repo.query(predicate="uses")} == {"f1", "f2"}

    repo.save_fact(fact("f1", "omnimemory", "uses", "official-mcp-sdk"))
    assert repo.query(object="fastmcp") == []
    assert [item.id for item in repo.query(object="official-mcp-sdk")] == ["f1"]

    assert repo.remove_fact("f2") is True
    assert repo.query(subject="persona") == []
    assert repo.query(object="fastapi") == []

    expired = fact("f3", "old", "uses", "expired")
    expired.meta["expire_at"] = 1
    repo.save_fact(expired)
    assert repo.gc_expired(now=2) == 1
    assert repo.query(subject="old") == []
    assert repo.get_fact("f3") is None


def test_graph_repo_entity_neighborhood_uses_subject_and_object_indexes():
    repo = GraphRepo()
    repo.save_fact(fact("f1", "omnimemory", "uses", "fastmcp"))
    repo.save_fact(fact("f2", "fastmcp", "supports", "sdk"))
    repo.save_fact(fact("f3", "persona", "uses", "fastapi"))

    facts = repo.query_entity_neighborhood(["omnimemory", "fastmcp"])

    assert {item.id for item in facts} == {"f1", "f2"}
