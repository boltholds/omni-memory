from __future__ import annotations

from app.builder import build_memory
from domain.models import DomainLink, DomainNode
from infra.embeddings.factory import HashEmbedder
from infra.repo.domain_graph_repo import DomainGraphRepo, domain_id


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_writeback_adds_scope_from_context_domains():
    memory = _memory()

    result = memory.write_items_raw(
        [
            {
                "id": "fact-scope-project-purpose",
                "type": "fact",
                "subject": "omni-memory",
                "predicate": "purpose",
                "object": "agent memory governance",
                "meta": {"confidence": 1.0},
            }
        ],
        source="codex-dev",
        meta={"domain_ids": ["domain:project:omni-memory", "domain:area:memory-hygiene"]},
    )

    assert result.saved_count == 1
    scope = result.saved[0].meta["scope"]
    assert scope["domain_ids"] == ["domain:project:omni-memory", "domain:area:memory-hygiene"]
    assert scope["environment"] == "dev"
    assert scope["durability"] == "durable"
    assert scope["exclude_from_consolidation"] is False


def test_writeback_quarantines_test_memory_as_ephemeral():
    memory = _memory()

    result = memory.write_items_raw(
        [
            {
                "id": "fact-test-fixture",
                "type": "fact",
                "subject": "alice-12345678",
                "predicate": "at",
                "object": "lighthouse",
                "meta": {"confidence": 1.0},
            }
        ],
        source="test",
    )

    assert result.saved_count == 1
    meta = result.saved[0].meta
    scope = meta["scope"]
    assert scope["environment"] == "test"
    assert scope["durability"] == "ephemeral"
    assert scope["exclude_from_consolidation"] is True
    assert meta["exclude_from_consolidation"] is True
    assert meta["volatility"] == "high"


def test_durable_memory_without_domain_gets_scope_warning():
    memory = _memory()

    result = memory.write_items_raw(
        [
            {
                "id": "fact-without-domain",
                "type": "fact",
                "subject": "domain hygiene",
                "predicate": "requires",
                "object": "explicit domain ids",
                "meta": {"confidence": 1.0},
            }
        ],
        source="codex-dev",
    )

    assert result.saved_count == 1
    assert "durable_memory_without_domain" in result.saved[0].meta["scope_warnings"]


def test_domain_graph_repo_links_projects_and_shared_domains():
    repo = DomainGraphRepo()
    omni = repo.upsert_node(DomainNode(id=domain_id("project", "omni-memory"), kind="project", name="OmniMemory"))
    persona = repo.upsert_node(DomainNode(id=domain_id("project", "persona-ai"), kind="project", name="Persona AI"))
    infra = repo.upsert_node(DomainNode(id=domain_id("knowledge_area", "infrastructure"), kind="knowledge_area", name="Infrastructure"))

    repo.add_link(DomainLink(source_id=omni.id, relation="has_subdomain", target_id=infra.id))
    repo.add_link(DomainLink(source_id=persona.id, relation="has_subdomain", target_id=infra.id))

    assert repo.count() == 3
    assert repo.link_count() == 2
    assert [node.id for node in repo.list_nodes(kind="project")] == [omni.id, persona.id]
    assert set(repo.related_domain_ids(infra.id)) == {omni.id, persona.id}
