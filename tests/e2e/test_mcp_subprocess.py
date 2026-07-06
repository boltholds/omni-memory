from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - exercised only when optional MCP SDK is absent.
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


pytestmark = pytest.mark.skipif(ClientSession is None, reason="MCP SDK is not installed")


@pytest.mark.asyncio
async def test_runtime_mcp_subprocess_writes_and_reloads_memory(tmp_path: Path):
    first = await _call_mcp(
        tmp_path,
        "omni_memory_finish_development_task",
        {
            "goal": "Smoke test MCP subprocess persistence",
            "summary": "Runtime MCP subprocess records a completed task.",
            "lesson": "The MCP stdio command should persist experience under the local .omni-memory directory.",
            "reuse_when": ["verifying MCP runtime packaging"],
            "run_distiller": False,
        },
    )

    assert first["experience"]["saved"] == 1

    second = await _call_mcp(
        tmp_path,
        "omni_memory_search_experiences",
        {"query": "MCP subprocess persistence", "k": 5},
    )

    assert any(
        item["goal"] == "Smoke test MCP subprocess persistence"
        for item in second["experiences"]
    )


async def _call_mcp(tmp_path: Path, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "omni_memory.runtime_cli", "mcp"],
        cwd=tmp_path,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert any(tool.name == tool_name for tool in tools.tools)
            result = await session.call_tool(tool_name, payload)
    return json.loads(result.content[0].text)
