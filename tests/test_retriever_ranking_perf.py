from __future__ import annotations

from app.retriever import Retriever, RetrievalScopeFilter
from domain.models import Fact, MemoryObject, Provenance
from infra.embeddings.factory import HashEmbedder
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo


def fact(fid: str, subject: str, object_: str) -> Fact:
    return Fact(id=fid, subject=subject, predicate="uses", object=object_, provenance=Provenance(source="test"), meta={"confidence": 1.0})


def note(nid: str, text: str, *, scope: dict | None = None, time: float = 0) -> MemoryObject:
    return MemoryObject(
        id=nid,
        type="note",
        payload={"text": text},
        provenance=Provenance(source="test", time=time),
        meta={"scope": scope or {}, "confidence": 1.0},
    )


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


def test_retriever_deduplicates_notes_by_content_and_keeps_scoped_copy():
    retriever = Retriever(VectorStoreRepo(embedder=HashEmbedder()), GraphRepo(), EpisodicRepo())
    items = [
        note("generic-new", "Simple note about MCP registry refactor", time=10),
        note(
            "scoped-old",
            "Simple note about MCP registry refactor",
            scope={"domain_ids": ["domain:project:omni-memory"], "durability": "durable"},
            time=1,
        ),
        note("other", "Different note about FastMCP schema required fields", time=5),
    ]

    ranked = retriever._rank_memory_items(
        "MCP registry refactor",
        items,
        {"domain:project:omni-memory": 4.0},
        RetrievalScopeFilter(domain_ids=["domain:project:omni-memory"]),
        memory_type="note",
        limit=5,
    )

    assert [item.id for item in ranked].count("generic-new") == 0
    assert [item.id for item in ranked].count("scoped-old") == 1
    assert len([item for item in ranked if item.payload["text"] == "Simple note about MCP registry refactor"]) == 1


def test_retriever_penalizes_unscoped_durable_memory_when_domain_context_exists():
    retriever = Retriever(VectorStoreRepo(embedder=HashEmbedder()), GraphRepo(), EpisodicRepo())
    items = [
        note("generic", "Generic durable memory", scope={"durability": "durable"}, time=99),
        note("scoped", "Domain-specific durable memory", scope={"domain_ids": ["domain:project:omni-memory"], "durability": "durable"}, time=1),
    ]

    ranked = retriever._rank_memory_items(
        "OmniMemory domain-specific memory",
        items,
        {"domain:project:omni-memory": 4.0},
        RetrievalScopeFilter(domain_ids=["domain:project:omni-memory"]),
        memory_type="note",
        limit=2,
    )

    assert [item.id for item in ranked] == ["scoped", "generic"]


def test_retriever_scope_filter_can_exclude_ephemeral_notes_without_hiding_normal_notes():
    retriever = Retriever(VectorStoreRepo(embedder=HashEmbedder()), GraphRepo(), EpisodicRepo())
    items = [
        note("bad", "Sandbox session note", scope={"environment": "sandbox", "durability": "session"}),
        note("good", "Durable domain note", scope={"domain_ids": ["domain:project:omni-memory"], "durability": "durable"}),
    ]

    ranked = retriever._rank_memory_items(
        "OmniMemory memory",
        items,
        {"domain:project:omni-memory": 4.0},
        RetrievalScopeFilter(domain_ids=["domain:project:omni-memory"], include_ephemeral=False),
        memory_type="note",
        limit=5,
    )

    assert [item.id for item in ranked] == ["good"]
