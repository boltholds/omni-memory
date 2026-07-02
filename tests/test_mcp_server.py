from __future__ import annotations

import json

from app.builder import build_memory
from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from app.mcp_server import StdioMcpServer
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _tool_text(response: dict) -> dict:
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


def test_mcp_tool_schemas_include_core_memory_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}

    assert "omni_memory_write_items" in names
    assert "omni_memory_retrieve" in names
    assert "omni_memory_context" in names
    assert "omni_memory_detect_conflicts" in names
    assert "omni_memory_session_commit" in names
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


def test_mcp_server_lists_and_calls_tools():
    server = StdioMcpServer(_memory())

    init = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "omni-memory"

    listed = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert any(tool["name"] == "omni_memory_stats" for tool in listed["result"]["tools"])

    called = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "omni_memory_stats", "arguments": {}},
        }
    )
    body = _tool_text(called)
    assert body["facts"] == 0
    assert body["llm_configured"] is False


def test_mcp_server_returns_json_rpc_error_for_unknown_tool():
    server = StdioMcpServer(_memory())

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "bad-tool",
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response["id"] == "bad-tool"
    assert response["error"]["code"] == -32000
    assert "Unknown tool" in response["error"]["message"]
