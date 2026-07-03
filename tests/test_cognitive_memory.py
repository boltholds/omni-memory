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
