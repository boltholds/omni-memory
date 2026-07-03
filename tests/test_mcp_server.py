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
        "session_turns": 1,
        "dry_run": True,
    }
    assert handlers["omni_memory_stats"]()["facts"] == 1

    cleared = handlers["omni_memory_clear"]()
    assert cleared == {
        "vector_objects": 1,
        "facts": 1,
        "episodes": 0,
        "session_turns": 1,
        "dry_run": False,
    }

    stats = handlers["omni_memory_stats"]()
    assert stats["vector_objects"] == 0
    assert stats["facts"] == 0
    assert stats["session_turns"] == 0
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["facts"] == []
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["semantic_chunks"] == []


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


@pytest.mark.asyncio
async def test_mcp_server_lists_and_calls_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_stats" for tool in listed)
    assert any(tool.name == "omni_memory_clear" for tool in listed)
    assert any(tool.name == "omni_memory_supersede_fact" for tool in listed)

    called = await server.call_tool("omni_memory_stats", {})
    body = _tool_text(called)
    assert body["facts"] == 0
    assert body["llm_configured"] is False


@pytest.mark.asyncio
async def test_mcp_server_rejects_unknown_tool():
    server = build_mcp_app(_memory())

    with pytest.raises(Exception, match="Unknown tool"):
        await server.call_tool("missing", {})
