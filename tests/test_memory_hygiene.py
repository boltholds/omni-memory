from __future__ import annotations

from omni_memory import build_memory
from omni_memory.domain.models import DomainLink, DomainNode
from omni_memory.infra.embeddings.factory import HashEmbedder
from omni_memory.infra.repo.domain_graph_repo import DomainGraphRepo, domain_id


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
    assert repo.successors(omni.id, relation="has_subdomain") == [infra.id]
    assert set(repo.predecessors(infra.id, relation="has_subdomain")) == {omni.id, persona.id}


def test_domain_graph_repo_supports_multihop_traversal():
    repo = DomainGraphRepo()
    omni = repo.upsert_node(DomainNode(id=domain_id("project", "omni-memory"), kind="project", name="OmniMemory"))
    infra = repo.upsert_node(DomainNode(id=domain_id("knowledge_area", "infrastructure"), kind="knowledge_area", name="Infrastructure"))
    ci = repo.upsert_node(DomainNode(id=domain_id("knowledge_area", "ci"), kind="knowledge_area", name="CI"))
    pytest = repo.upsert_node(DomainNode(id=domain_id("artifact_group", "pytest"), kind="artifact_group", name="pytest"))

    repo.add_link(DomainLink(source_id=omni.id, relation="has_subdomain", target_id=infra.id))
    repo.add_link(DomainLink(source_id=infra.id, relation="has_subdomain", target_id=ci.id))
    repo.add_link(DomainLink(source_id=ci.id, relation="depends_on", target_id=pytest.id))

    assert repo.successors(infra.id) == [ci.id]
    assert repo.predecessors(ci.id) == [infra.id]
    assert repo.reachable_domain_ids(omni.id, max_depth=1) == [infra.id]

    reachable = repo.reachable_domain_ids(omni.id, max_depth=3)
    assert set(reachable) == {infra.id, ci.id, pytest.id}
    assert len(reachable) == 3


def test_domain_aware_retrieval_prefers_matching_project_scope():
    memory = _memory()
    omni = memory.domain_graph_repo.upsert_node(DomainNode(id=domain_id("project", "omni-memory"), kind="project", name="OmniMemory"))
    persona = memory.domain_graph_repo.upsert_node(DomainNode(id=domain_id("project", "persona-ai"), kind="project", name="Persona AI"))

    memory.write_skill(
        name="Fix dependency issue",
        problem="Persona dependency issue in integration tests",
        procedure=["Inspect Persona dependencies"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        meta={"domain_ids": [persona.id]},
        source="codex-dev",
    )
    memory.write_skill(
        name="Fix dependency issue",
        problem="OmniMemory dependency issue in CI collection",
        procedure=["Inspect OmniMemory imports"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        meta={"domain_ids": [omni.id]},
        source="codex-dev",
    )

    result = memory.retrieve("OmniMemory dependency issue", intent="write_code", k_eps=2)

    assert len(result.skills) == 2
    assert result.skills[0].meta["scope"]["domain_ids"] == [omni.id]


def test_domain_aware_retrieval_downranks_test_ephemeral_memory():
    memory = _memory()

    memory.write_skill(
        name="Fix dependency issue",
        problem="Test fixture dependency issue",
        procedure=["Use test-only workaround"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        source="test",
    )
    memory.write_skill(
        name="Fix dependency issue",
        problem="Durable project dependency issue",
        procedure=["Use durable project fix"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        meta={"domain_ids": ["domain:project:omni-memory"]},
        source="codex-dev",
    )

    result = memory.retrieve("dependency issue", intent="write_code", k_eps=2)

    assert len(result.skills) == 2
    assert result.skills[0].meta["scope"]["durability"] == "durable"
    assert result.skills[-1].meta["scope"]["environment"] == "test"


def test_scope_retrieval_strict_domain_filter_keeps_only_requested_domain():
    memory = _memory()
    omni = memory.domain_graph_repo.upsert_node(DomainNode(id=domain_id("project", "omni-memory"), kind="project", name="OmniMemory"))
    persona = memory.domain_graph_repo.upsert_node(DomainNode(id=domain_id("project", "persona-ai"), kind="project", name="Persona AI"))

    memory.write_skill(
        name="Fix dependency issue",
        problem="Persona dependency issue",
        procedure=["Inspect Persona dependencies"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        meta={"domain_ids": [persona.id]},
        source="codex-dev",
    )
    memory.write_skill(
        name="Fix dependency issue",
        problem="OmniMemory dependency issue",
        procedure=["Inspect OmniMemory dependencies"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        meta={"domain_ids": [omni.id]},
        source="codex-dev",
    )

    result = memory.retrieve(
        "dependency issue",
        intent="write_code",
        k_eps=5,
        scope={"domain_ids": [omni.id], "strict_domains": True},
    )

    assert [skill.meta["scope"]["domain_ids"] for skill in result.skills] == [[omni.id]]


def test_scope_retrieval_environment_and_ephemeral_filters():
    memory = _memory()

    memory.write_skill(
        name="Fix dependency issue",
        problem="Test dependency issue",
        procedure=["Use test-only workaround"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        source="test",
    )
    memory.write_skill(
        name="Fix dependency issue",
        problem="Durable dependency issue",
        procedure=["Use durable fix"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        source="codex-dev",
    )

    dev_only = memory.retrieve(
        "dependency issue",
        intent="write_code",
        k_eps=5,
        scope={"environments": ["dev"]},
    )
    no_ephemeral = memory.retrieve(
        "dependency issue",
        intent="write_code",
        k_eps=5,
        scope={"include_ephemeral": False},
    )

    assert len(dev_only.skills) == 1
    assert dev_only.skills[0].meta["scope"]["environment"] == "dev"
    assert len(no_ephemeral.skills) == 1
    assert no_ephemeral.skills[0].meta["scope"]["durability"] == "durable"


def test_scope_retrieval_memory_type_filter_disables_unrequested_channels():
    memory = _memory()

    memory.write_skill(
        name="Fix dependency issue",
        problem="Dependency issue",
        procedure=["Use durable fix"],
        reuse_when=["dependency issue"],
        confidence=0.9,
        source="codex-dev",
    )
    memory.write_note("Dependency issue notes should be hidden when only skills are requested.", source="codex-dev")

    result = memory.retrieve(
        "dependency issue",
        intent="write_code",
        k_eps=5,
        k_sem=5,
        scope={"memory_types": ["skill"]},
    )

    assert result.skills
    assert result.semantic_chunks == []
    assert result.facts == []
    assert result.experiences == []


def test_consolidation_excludes_test_and_ephemeral_experiences():
    memory = _memory()

    for idx in range(2):
        memory.record_experience(
            goal="Consolidate durable domain experience",
            context="Durable project experience",
            decision="Promote durable project lesson",
            actions=["Apply durable fix"],
            outcome=f"durable success {idx}",
            evaluation={"success": True, "tests": "passed"},
            lesson="Durable project lessons may become skills.",
            reuse_when=["durable project lesson"],
            confidence=0.92,
            meta={"domain_ids": ["domain:project:omni-memory"]},
            source="codex-dev",
        )
        memory.record_experience(
            goal="Consolidate test fixture experience",
            context="Ephemeral test fixture experience",
            decision="Do not promote test fixture lesson",
            actions=["Apply test workaround"],
            outcome=f"test success {idx}",
            evaluation={"success": True, "tests": "passed"},
            lesson="Test fixture lessons should not become skills.",
            reuse_when=["test fixture lesson"],
            confidence=0.99,
            source="test",
        )

    result = memory.consolidate_experiences(dry_run=False, min_confidence=0.85)

    assert result.saved_skills
    skill_names = [skill.name for skill in result.saved_skills]
    assert any("durable" in name.lower() or "promote durable" in name.lower() for name in skill_names)
    assert all("test fixture" not in skill.name.lower() for skill in result.saved_skills)
    assert all("test" not in evidence_id for skill in result.saved_skills for evidence_id in skill.evidence_ids)
