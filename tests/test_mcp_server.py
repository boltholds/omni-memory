from __future__ import annotations

import json

import pytest

from app.builder import build_memory
from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from app.mcp_server import build_mcp_app
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_mcp_tool_schemas_include_core_memory_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}

    assert "omni_memory_write_items" in names
    assert "omni_memory_retrieve" in names
    assert "omni_memory_context" in names
    assert "omni_memory_detect_conflicts" in names
    assert "omni_memory_session_commit" in names
    assert "omni_memory_clear" in names
    assert "omni_memory_list_facts" in names
    assert "omni_memory_get_fact" in names
    assert "omni_memory_retract_fact" in names
    assert "omni_memory_supersede_fact" in names
    assert "omni_memory_write_decision" in names
    assert "omni_memory_list_decisions" in names
    assert "omni_memory_get_decision" in names
    assert "omni_memory_write_experience" in names
    assert "omni_memory_search_experiences" in names
    assert "omni_memory_record_agent_cycle" in names
    assert "omni_memory_stats" in names


def test_mcp_handlers_write_retrieve_context_and_conflicts():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    assert write["saved"] == 1

    retrieve = handlers["omni_memory_retrieve"](query="Where is Alice?")
    assert any(fact["object"] == "lighthouse" for fact in retrieve["facts"])

    context = handlers["omni_memory_context"](query="Where is Alice?")
    assert any(section["title"] in {"Facts", "Current Beliefs"} for section in context["sections"])

    conflicts = handlers["omni_memory_detect_conflicts"](
        facts=[
            {"id": "f1", "subject": "alice", "predicate": "at", "object": "lighthouse"},
            {"id": "f2", "subject": "alice", "predicate": "at", "object": "bridge"},
        ]
    )
    assert conflicts["conflicts"][0]["key"] == "alice::at"


def test_mcp_clear_removes_selected_memory_stores():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    handlers["omni_memory_write_note"](text="Project uses MCP tools.", source="test")
    handlers["omni_memory_session_ingest_turn"](role="user", content="remember this transiently")

    dry_run = handlers["omni_memory_clear"](dry_run=True)
    assert dry_run == {
        "vector_objects": 1,
        "facts": 1,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "session_turns": 1,
        "dry_run": True,
    }
    assert handlers["omni_memory_stats"]()["facts"] == 1

    cleared = handlers["omni_memory_clear"]()
    assert cleared == {
        "vector_objects": 1,
        "facts": 1,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "session_turns": 1,
        "dry_run": False,
    }

    stats = handlers["omni_memory_stats"]()
    assert stats["vector_objects"] == 0
    assert stats["facts"] == 0
    assert stats["decisions"] == 0
    assert stats["experiences"] == 0
    assert stats["session_turns"] == 0
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["facts"] == []
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["semantic_chunks"] == []


def test_mcp_clear_preserves_excluded_memory_stores():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    handlers["omni_memory_write_note"](text="Project uses MCP tools.", source="test")
    handlers["omni_memory_session_ingest_turn"](role="user", content="transient turn")

    cleared = handlers["omni_memory_clear"](
        include_facts=False,
        include_episodes=False,
        include_decisions=False,
        include_experiences=False,
    )
    assert cleared == {
        "vector_objects": 1,
        "facts": 0,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "session_turns": 1,
        "dry_run": False,
    }

    stats = handlers["omni_memory_stats"]()
    assert stats["vector_objects"] == 0
    assert stats["facts"] == 1
    assert stats["session_turns"] == 0

    retrieved = handlers["omni_memory_retrieve"](query="Where is Alice?")
    assert any(fact["object"] == "lighthouse" for fact in retrieved["facts"])
    assert handlers["omni_memory_retrieve"](query="Project uses MCP tools")["semantic_chunks"] == []


def test_mcp_fact_maintenance_supersedes_without_creating_current_conflict():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_items"](
        items=[
            {
                "id": "fact-mcp-old",
                "type": "fact",
                "subject": "mcp_server",
                "predicate": "implemented_with",
                "object": "minimal json-rpc",
                "meta": {"confidence": 0.7},
            }
        ],
        source="test",
    )
    assert write["saved"] == 1

    supersede = handlers["omni_memory_supersede_fact"](
        fact_id="fact-mcp-old",
        new_fact={
            "id": "fact-mcp-new",
            "object": "official MCP SDK FastMCP",
            "meta": {"confidence": 0.99},
        },
        reason="Migrated to official MCP SDK",
        source="test",
    )
    assert supersede["applied"] is True
    assert supersede["fact"]["id"] == "fact-mcp-new"

    old = handlers["omni_memory_get_fact"](fact_id="fact-mcp-old")["fact"]
    assert old["meta"]["status"] == "historical"
    assert old["meta"]["superseded_by"] == "fact-mcp-new"

    current = handlers["omni_memory_list_facts"](status="current")["facts"]
    historical = handlers["omni_memory_list_facts"](status="historical")["facts"]
    assert [fact["id"] for fact in current] == ["fact-mcp-new"]
    assert [fact["id"] for fact in historical] == ["fact-mcp-old"]

    all_facts = handlers["omni_memory_list_facts"]()["facts"]
    conflicts = handlers["omni_memory_detect_conflicts"](facts=all_facts)
    assert conflicts["conflicts"] == []

    retrieved = handlers["omni_memory_retrieve"](query="mcp_server implementation", k_sem=1)
    belief = retrieved["beliefs"][0]
    assert belief["current"]["id"] == "fact-mcp-new"
    assert [fact["id"] for fact in belief["historical"]] == ["fact-mcp-old"]


def test_mcp_fact_maintenance_retracts_fact_from_current_beliefs():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_items"](
        items=[
            {
                "id": "fact-temp",
                "type": "fact",
                "subject": "temporary_fact",
                "predicate": "status",
                "object": "active",
            }
        ],
        source="test",
    )

    retracted = handlers["omni_memory_retract_fact"](
        fact_id="fact-temp",
        reason="No longer true",
    )
    assert retracted["applied"] is True
    assert retracted["fact"]["meta"]["status"] == "retracted"

    retrieved = handlers["omni_memory_retrieve"](query="temporary_fact status", k_sem=1)
    assert retrieved["beliefs"] == []


def test_mcp_decision_records_are_written_listed_and_retrieved_in_context():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_decision"](
        title="Use FastMCP for MCP server",
        decision="Use official MCP SDK FastMCP instead of a handwritten JSON-RPC loop.",
        context="MCP clients expect protocol-compatible stdio behavior.",
        consequences=["Better compatibility with Codex and other MCP clients."],
        alternatives=["Keep minimal newline-delimited JSON-RPC server."],
        refs={"files": ["app/mcp_server.py"]},
        source="test",
    )
    assert write["saved"] == 1

    listed = handlers["omni_memory_list_decisions"](limit=5)["decisions"]
    assert listed[0]["title"] == "Use FastMCP for MCP server"

    decision_id = listed[0]["id"]
    got = handlers["omni_memory_get_decision"](decision_id=decision_id)["decision"]
    assert got["decision"].startswith("Use official MCP SDK")

    context = handlers["omni_memory_context"](query="Why did we choose FastMCP?")
    sections = {section["title"]: section["body"] for section in context["sections"]}
    assert "Decision Records" in sections
    assert "Use FastMCP for MCP server" in sections["Decision Records"]


def test_mcp_experience_records_are_searchable_and_retrieved_in_context():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_experience"](
        goal="Improve MCP compatibility",
        decision="Use official protocol SDK",
        actions=["Replaced handwritten JSON-RPC loop with FastMCP"],
        outcome="Codex could discover MCP tools after restart.",
        evaluation={"tests": "passed", "success": True},
        lesson="Prefer the official protocol SDK when MCP client compatibility matters.",
        reuse_when=["building MCP integrations", "protocol compatibility matters"],
        avoid_when=["dependency footprint must stay minimal"],
        confidence=0.95,
        refs={"files": ["app/mcp_server.py"]},
        source="test",
    )
    assert write["saved"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="MCP compatibility protocol SDK",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Improve MCP compatibility"
    assert "protocol SDK" in found[0]["lesson"]

    context = handlers["omni_memory_context"](query="How should we improve MCP compatibility?")
    sections = {section["title"]: section["body"] for section in context["sections"]}
    assert "Relevant Experience" in sections
    assert "Prefer the official protocol SDK" in sections["Relevant Experience"]
    assert "building MCP integrations" in sections["Relevant Experience"]


def test_mcp_agent_cycle_records_reusable_experience():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_record_agent_cycle"](
        goal="Make fact maintenance manageable",
        plan=["Keep repositories as storage primitives", "Move semantic operations into strategies"],
        decisions=["Use FactMaintenanceService with strategies"],
        actions=["Added list/get/retract/supersede strategies", "Exposed MCP maintenance tools"],
        outcome="Fact maintenance works without making GraphRepo responsible for business semantics.",
        tests=["71 passed"],
        files=["app/fact_maintenance.py", "infra/repo/graph_repo.py"],
        side_effects=["MCP server must be restarted for new tools"],
        lesson="Maintenance behavior belongs in strategies, not storage repositories.",
        reuse_when=["adding semantic operations over persisted memory"],
        avoid_when=["operation is a simple storage primitive"],
        confidence=0.96,
        source="test",
    )
    assert write["saved"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="semantic operations persisted memory strategies",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Make fact maintenance manageable"
    assert found[0]["decision"] == "Use FactMaintenanceService with strategies"
    assert found[0]["evaluation"]["tests"] == ["71 passed"]
    assert found[0]["refs"]["files"] == ["app/fact_maintenance.py", "infra/repo/graph_repo.py"]
    assert found[0]["meta"]["recorded_from"] == "agent_cycle"


@pytest.mark.asyncio
async def test_mcp_server_lists_and_calls_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_stats" for tool in listed)
    assert any(tool.name == "omni_memory_clear" for tool in listed)
    assert any(tool.name == "omni_memory_supersede_fact" for tool in listed)
    assert any(tool.name == "omni_memory_write_decision" for tool in listed)
    assert any(tool.name == "omni_memory_write_experience" for tool in listed)
    assert any(tool.name == "omni_memory_record_agent_cycle" for tool in listed)

    called = await server.call_tool("omni_memory_stats", {})
    body = _tool_text(called)
    assert body["facts"] == 0
    assert body["llm_configured"] is False


@pytest.mark.asyncio
async def test_mcp_server_rejects_unknown_tool():
    server = build_mcp_app(_memory())

    with pytest.raises(Exception, match="Unknown tool"):
        await server.call_tool("missing", {})
