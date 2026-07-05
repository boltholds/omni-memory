from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omni_memory import build_memory
from omni_memory.domain.models import Fact
from omni_memory.infra.graph_backend import GraphEdge, NetworkxGraphBackend
from omni_memory.infra.repo.graph_repo import GraphRepo


class RecordingGraphBackend:
    def __init__(self) -> None:
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.upsert_calls: list[tuple[str, str, str]] = []
        self.remove_calls: list[tuple[str, str, str]] = []
        self.clear_calls = 0

    def edge_count(self) -> int:
        return len(self.edges)

    def clear(self) -> int:
        removed = len(self.edges)
        self.edges.clear()
        self.clear_calls += 1
        return removed

    def upsert_edge(self, subject: str, object_: str, key: str, attrs: dict[str, Any]) -> None:
        self.upsert_calls.append((subject, object_, key))
        self.edges[(subject, object_, key)] = dict(attrs)

    def get_edge(self, subject: str, object_: str, key: str) -> dict[str, Any] | None:
        item = self.edges.get((subject, object_, key))
        return dict(item) if item is not None else None

    def remove_edge(self, subject: str, object_: str, key: str) -> bool:
        self.remove_calls.append((subject, object_, key))
        return self.edges.pop((subject, object_, key), None) is not None

    def iter_edges(self) -> Iterable[GraphEdge]:
        for (subject, object_, key), attrs in self.edges.items():
            yield subject, object_, key, dict(attrs)


def _fact(fact_id: str, subject: str, predicate: str, object_: str, **meta: Any) -> Fact:
    return Fact(id=fact_id, subject=subject, predicate=predicate, object=object_, meta=meta)


def test_graph_repo_uses_injected_backend_for_fact_storage():
    backend = RecordingGraphBackend()
    repo = GraphRepo(backend=backend)

    repo.save_fact(_fact("f1", "alice", "at", "lighthouse"))
    repo.save_fact(_fact("f2", "alice", "likes", "tea"))

    assert backend.upsert_calls == [
        ("alice", "lighthouse", "f1"),
        ("alice", "tea", "f2"),
    ]
    assert repo.count() == 2
    assert repo.get_fact("f1").object == "lighthouse"
    assert [fact.id for fact in repo.query(subject="alice")] == ["f1", "f2"]
    assert [fact.id for fact in repo.query(predicate="likes")] == ["f2"]


def test_graph_repo_reindexes_when_fact_endpoint_changes():
    backend = RecordingGraphBackend()
    repo = GraphRepo(backend=backend)

    repo.save_fact(_fact("f1", "alice", "at", "lighthouse"))
    repo.save_fact(_fact("f1", "alice", "at", "bridge"))

    assert backend.remove_calls == [("alice", "lighthouse", "f1")]
    assert repo.get_fact("f1").object == "bridge"
    assert repo.query(object="lighthouse") == []
    assert [fact.id for fact in repo.query(object="bridge")] == ["f1"]


def test_graph_repo_gc_expired_uses_backend_iteration():
    backend = RecordingGraphBackend()
    repo = GraphRepo(backend=backend)

    repo.save_fact(_fact("fresh", "alice", "at", "lighthouse", expire_at=20.0))
    repo.save_fact(_fact("old", "bob", "at", "bridge", expire_at=1.0))

    removed = repo.gc_expired(now=10.0)

    assert removed == 1
    assert repo.get_fact("old") is None
    assert repo.get_fact("fresh") is not None


def test_networkx_graph_backend_matches_graph_backend_contract():
    repo = GraphRepo(backend=NetworkxGraphBackend())

    repo.save_fact(_fact("f1", "alice", "at", "lighthouse"))

    assert repo.count() == 1
    assert repo.get_fact("f1").object == "lighthouse"
    assert [fact.id for fact in repo.query(subject="alice")] == ["f1"]


def test_build_memory_accepts_graph_backend_prototype():
    backend = RecordingGraphBackend()
    memory = build_memory(graph_backend=backend)

    memory.write_fact("alice", "at", "lighthouse", source="test")
    facts = memory.retrieve("Where is Alice?", k_sem=0).facts

    assert backend.edge_count() == 1
    assert any(fact.object == "lighthouse" for fact in facts)
