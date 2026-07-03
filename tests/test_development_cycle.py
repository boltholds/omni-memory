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


def test_mcp_development_cycle_draft_does_not_write_memory():
    handlers = build_mcp_handlers(_memory())

    draft = handlers["omni_memory_draft_development_cycle"](
        goal="Wire development cycle recorder",
        summary="Added draft and record tools.",
        changed_files=["app/development_cycle.py", "app/integrations/mcp.py"],
        commands_run=["poetry run pytest -q tests/test_development_cycle.py"],
        tests=["development cycle tests passed"],
        decisions=["Draft first, record only when lesson is explicit"],
        reuse_when=["capturing completed development work"],
    )

    assert draft["goal"] == "Wire development cycle recorder"
    assert "Ran command: poetry run pytest" in draft["actions"][1]
    assert draft["files"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert draft["meta"]["recorded_from"] == "development_cycle"
    assert draft["meta"]["draft"] is True
    assert handlers["omni_memory_stats"]()["experiences"] == 0


def test_mcp_development_cycle_record_writes_searchable_experience():
    handlers = build_mcp_handlers(_memory())

    written = handlers["omni_memory_record_development_cycle"](
        goal="Wire development cycle recorder",
        summary="Added controlled dev-cycle recording.",
        changed_files=["app/development_cycle.py", "app/integrations/mcp.py"],
        commands_run=["poetry run pytest -q"],
        tests=["83 passed"],
        decisions=["Use explicit lesson before recording"],
        outcome="Development cycle recorder is available through MCP.",
        lesson="Development cycles should be drafted first, then recorded as experience after a reusable lesson is explicit.",
        reuse_when=["recording meaningful development outcomes"],
        source="test",
    )

    assert written["saved"] == 1
    assert handlers["omni_memory_stats"]()["experiences"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="development cycles drafted recorded reusable lesson",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Wire development cycle recorder"
    assert found[0]["refs"]["files"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert "poetry run pytest -q" in found[0]["actions"][1]
    assert found[0]["evaluation"]["tests"] == ["83 passed"]
    assert found[0]["meta"]["recorded_from"] == "development_cycle"


def test_mcp_schemas_and_fastmcp_include_development_cycle_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}
    assert "omni_memory_draft_development_cycle" in names
    assert "omni_memory_record_development_cycle" in names


@pytest.mark.asyncio
async def test_fastmcp_lists_development_cycle_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_draft_development_cycle" for tool in listed)
    assert any(tool.name == "omni_memory_record_development_cycle" for tool in listed)

    called = await server.call_tool(
        "omni_memory_draft_development_cycle",
        {"goal": "Check FastMCP dev-cycle tool"},
    )
    body = _tool_text(called)
    assert body["goal"] == "Check FastMCP dev-cycle tool"
