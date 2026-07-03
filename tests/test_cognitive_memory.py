from __future__ import annotations

from app.builder import build_memory
from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_mcp_schemas_include_cognitive_memory_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}

    assert "omni_memory_write_skill" in names
    assert "omni_memory_list_skills" in names
    assert "omni_memory_get_skill" in names
    assert "omni_memory_search_skills" in names
    assert "omni_memory_write_failure_pattern" in names
    assert "omni_memory_list_failure_patterns" in names
    assert "omni_memory_get_failure_pattern" in names
    assert "omni_memory_search_failure_patterns" in names
    assert "omni_memory_consolidate_experiences" in names


def test_skill_memory_write_list_get_and_search():
    handlers = build_mcp_handlers(_memory())

    written = handlers["omni_memory_write_skill"](
        name="Fix CI dependency issue",
        problem="A test import needs a package that is not available.",
        procedure=["Read the import error", "Remove the unused dependency", "Keep a regression test"],
        reuse_when=["test collection fails", "missing package"],
        confidence=0.9,
        source="test",
    )

    skill_id = written["skill"]["id"]
    assert written["saved"] == 1

    listed = handlers["omni_memory_list_skills"]()
    assert listed["skills"][0]["id"] == skill_id

    found = handlers["omni_memory_get_skill"](skill_id=skill_id)
    assert found["skill"]["id"] == skill_id

    searched = handlers["omni_memory_search_skills"](query="dependency package test", k=5)
    assert searched["skills"][0]["id"] == skill_id


def test_failure_pattern_memory_write_list_get_and_search():
    handlers = build_mcp_handlers(_memory())

    written = handlers["omni_memory_write_failure_pattern"](
        symptom="Tests fail during collection",
        root_cause="A module imports a package that is not installed.",
        fix="Remove the unused import or add the dependency explicitly.",
        detection="Collection fails before test bodies run.",
        evidence_ids=["run_001"],
        confidence=0.9,
        source="test",
    )

    pattern_id = written["failure_pattern"]["id"]
    assert written["saved"] == 1

    listed = handlers["omni_memory_list_failure_patterns"]()
    assert listed["failure_patterns"][0]["id"] == pattern_id

    found = handlers["omni_memory_get_failure_pattern"](pattern_id=pattern_id)
    assert found["failure_pattern"]["id"] == pattern_id

    searched = handlers["omni_memory_search_failure_patterns"](query="collection package", k=5)
    assert searched["failure_patterns"][0]["id"] == pattern_id


def test_cognitive_memory_writes_go_through_writeback_policies():
    handlers = build_mcp_handlers(_memory())

    skill = handlers["omni_memory_write_skill"](
        name="Contact maintainer at root@example.com",
        confidence=0.9,
        source="test",
    )
    pattern = handlers["omni_memory_write_failure_pattern"](
        symptom="Failure includes token=abcdabcdabcdabcd",
        confidence=0.9,
        source="test",
    )

    assert skill["saved"] == 0
    assert skill["skill"] is None
    assert any("pii_email_blocked" in reason for reason in skill["reasons"])

    assert pattern["saved"] == 0
    assert pattern["failure_pattern"] is None
    assert any("pii_secret_blocked" in reason for reason in pattern["reasons"])

    stats = handlers["omni_memory_stats"]()
    assert stats["skills"] == 0
    assert stats["failure_patterns"] == 0


def test_stats_and_clear_include_cognitive_memory_repos():
    memory = _memory()
    handlers = build_mcp_handlers(memory)

    handlers["omni_memory_write_skill"](name="Reusable skill", confidence=0.8)
    handlers["omni_memory_write_failure_pattern"](symptom="Reusable failure", confidence=0.8)

    stats = handlers["omni_memory_stats"]()
    assert stats["skills"] == 1
    assert stats["failure_patterns"] == 1

    memory.clear(
        include_vectors=False,
        include_facts=False,
        include_episodes=False,
        include_decisions=False,
        include_experiences=False,
    )
    stats_after = handlers["omni_memory_stats"]()
    assert stats_after["skills"] == 0
    assert stats_after["failure_patterns"] == 0


def test_consolidation_dry_run_proposes_skill_without_writing():
    handlers = build_mcp_handlers(_memory())

    for suffix in ["one", "two"]:
        handlers["omni_memory_write_experience"](
            goal="Fix CI dependency issue",
            context="CI collection fails because a package import is missing.",
            decision="Remove unnecessary dependency from answer chain",
            actions=["Read failing import", "Replace dependency with local runner"],
            outcome=f"pytest passed after fix {suffix}",
            evaluation={"success": True, "tests": "passed"},
            lesson="Prefer removing unnecessary dependency before adding a heavy package.",
            reuse_when=["CI fails during collection", "missing package import"],
            confidence=0.92,
            source="codex-dev",
            meta={"domain_ids": ["domain:project:omni-memory", "domain:area:ci"]},
        )

    result = handlers["omni_memory_consolidate_experiences"](dry_run=True, min_confidence=0.85)

    assert result["dry_run"] is True
    assert any(proposal["kind"] == "skill" for proposal in result["proposals"])
    assert handlers["omni_memory_stats"]()["skills"] == 0


def test_consolidation_apply_saves_skill_and_failure_pattern_for_context():
    handlers = build_mcp_handlers(_memory())

    for suffix in ["one", "two"]:
        handlers["omni_memory_write_experience"](
            goal="Fix CI dependency issue",
            context="CI collection fails because a package import is missing.",
            decision="Remove unnecessary dependency from answer chain",
            actions=["Read failing import", "Replace dependency with local runner"],
            outcome=f"pytest passed after fix {suffix}",
            evaluation={"success": True, "tests": "passed"},
            lesson="Prefer removing unnecessary dependency before adding a heavy package.",
            reuse_when=["CI fails during collection", "missing package import"],
            confidence=0.92,
            source="codex-dev",
            meta={"domain_ids": ["domain:project:omni-memory", "domain:area:ci"]},
        )

    handlers["omni_memory_write_experience"](
        goal="Fix CI dependency issue",
        context="CI collection fails because a package import is missing.",
        decision="Add missing dependency blindly",
        actions=["Added package without checking if import was necessary"],
        outcome="pytest failed with regression during collection",
        evaluation={"success": False, "tests": "failed"},
        lesson="Blindly adding dependencies can create CI regressions.",
        reuse_when=["CI fails during collection", "missing package import"],
        confidence=0.91,
        source="codex-dev",
        meta={"domain_ids": ["domain:project:omni-memory", "domain:area:ci"]},
    )

    applied = handlers["omni_memory_consolidate_experiences"](dry_run=False, min_confidence=0.85)

    assert applied["dry_run"] is False
    assert applied["saved_skills"]
    assert applied["saved_failure_patterns"]
    assert handlers["omni_memory_stats"]()["skills"] >= 1
    assert handlers["omni_memory_stats"]()["failure_patterns"] >= 1

    context = handlers["omni_memory_context"](
        query="CI dependency issue missing package collection failure",
        intent="debug_failure",
    )
    sections = {section["title"]: section["body"] for section in context["sections"]}
    assert "Skills" in sections
    assert "Failure Patterns" in sections
    assert "dependency" in sections["Skills"].lower()
    assert "pytest failed" in sections["Failure Patterns"].lower()
