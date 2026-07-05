from __future__ import annotations

import json

import pytest

from omni_memory import build_memory
from omni_memory.fact_mining import StaticFactExtractor
from omni_memory.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from omni_memory.mcp_server import build_mcp_app
from omni_memory.infra.embeddings.factory import HashEmbedder


TEXT = "OmniMemory uses FastMCP for the MCP server."


def _memory():
    return build_memory(
        use_llm=False,
        embedder=HashEmbedder(),
        fact_extractor=StaticFactExtractor(
            [
                {
                    "subject": "OmniMemory",
                    "predicate": "uses",
                    "object": "FastMCP",
                    "confidence": 0.92,
                    "evidence_quote": "OmniMemory uses FastMCP for the MCP server.",
                    "reason": "Explicit project implementation fact.",
                    "temporal_scope": "current",
                }
            ]
        ),
    )


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_mcp_schema_includes_fact_mining_tool():
    schema = next(tool for tool in MCP_TOOL_SCHEMAS if tool["name"] == "omni_memory_mine_facts")
    props = schema["inputSchema"]["properties"]

    assert "text" in props
    assert props["dry_run"]["default"] is True
    assert props["policy_mode"]["default"] == "review"
    assert "domain_ids" in props


def test_mcp_fact_mining_handler_dry_run_does_not_write():
    memory = _memory()
    handlers = build_mcp_handlers(memory)

    result = handlers["omni_memory_mine_facts"](
        text=TEXT,
        source="mcp-test",
        dry_run=True,
        domain_ids=["domain:project:omni-memory"],
    )

    assert result["dry_run"] is True
    assert result["candidate_count"] == 1
    assert result["accepted_count"] == 1
    assert result["saved_count"] == 0
    assert result["candidates"][0]["status"] == "policy_accepted"
    assert memory.repository_stats()["facts"] == 0


def test_mcp_fact_mining_handler_apply_writes_fact():
    memory = _memory()
    handlers = build_mcp_handlers(memory)

    result = handlers["omni_memory_mine_facts"](
        text=TEXT,
        source="mcp-test",
        dry_run=False,
    )

    assert result["dry_run"] is False
    assert result["saved_count"] == 1
    assert result["candidates"][0]["status"] == "saved"
    assert memory.repository_stats()["facts"] == 1


@pytest.mark.asyncio
async def test_fastmcp_server_lists_and_calls_fact_mining_tool():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_mine_facts" for tool in listed)

    called = await server.call_tool(
        "omni_memory_mine_facts",
        {"text": TEXT, "source": "mcp-test", "dry_run": True},
    )
    body = _tool_text(called)
    assert body["candidate_count"] == 1
    assert body["candidates"][0]["status"] == "policy_accepted"
