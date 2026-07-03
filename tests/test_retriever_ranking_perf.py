from __future__ import annotations

from app.retriever import Retriever, RetrievalScopeFilter
from domain.models import Fact, Provenance
from infra.embeddings.factory import HashEmbedder
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo


def fact(fid: str, subject: str, object_: str) -> Fact:
    return Fact(id=fid, subject=subject, predicate="uses", object=object_, provenance=Provenance(source="test"), meta={"confidence": 1.0})


class CountingGraphRepo(GraphRepo):
    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0
        self.neighborhood_calls = 0

    def query(self, **query_spec):
        self.query_calls += 1
        return super().query(**query_spec)

    def query_entity_neighborhood(self, entities, *, include_incoming=True, include_outgoing=True):
        self.neighborhood_calls += 1
        return super().query_entity_neighborhood(entities, include_incoming=include_incoming, include_outgoing=include_outgoing)


def test_retriever_graph_expansion_uses_batched_neighborhood_queries():
    graph = CountingGraphRepo()
    graph.save_fact(fact("f1", "omnimemory", "fastmcp"))
    graph.save_fact(fact("f2", "fastmcp", "sdk"))
    for idx in range(40):
        graph.save_fact(fact(f"noise-{idx}", f"noise-{idx}", f"other-{idx}"))

    retriever = Retriever(VectorStoreRepo(embedder=HashEmbedder()), graph, EpisodicRepo())
    facts = retriever._expand_graph_two_hop(["omnimemory"])

    assert {item.id for item in facts} >= {"f1", "f2"}
    assert graph.neighborhood_calls <= 2
    assert graph.query_calls == 0


def test_retriever_rank_top_k_keeps_expected_ordering():
    retriever = Retriever(VectorStoreRepo(embedder=HashEmbedder()), GraphRepo(), EpisodicRepo())
    items = [fact("old", "a", "old"), fact("new", "a", "new"), fact("other", "b", "other")]
    items[0].provenance.time = 1
    items[1].provenance.time = 3
    items[2].provenance.time = 2
    items[0].meta["scope"] = {"domain_ids": ["domain:project:omni-memory"]}
    items[1].meta["scope"] = {"domain_ids": ["domain:project:omni-memory"]}

    ranked = retriever._rank_memory_items(
        "dependency query",
        items,
        {"domain:project:omni-memory": 4.0},
        RetrievalScopeFilter(),
        memory_type="fact",
        limit=2,
    )

    assert [item.id for item in ranked] == ["new", "old"]
