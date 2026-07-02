from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from app.memory import OmniMemory


class StdioMcpServer:
    """Minimal MCP-compatible JSON-RPC server over newline-delimited stdio."""

    def __init__(self, memory: OmniMemory) -> None:
        self._handlers = build_mcp_handlers(memory)

    def serve(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout

        for raw in stdin:
            raw = raw.strip()
            if not raw:
                continue

            response = self.handle_message(json.loads(raw))
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")

        try:
            result = self._dispatch(method, message.get("params") or {})
        except Exception as exc:
            if request_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": f"{type(exc).__name__}: {exc}",
                },
            }

        if request_id is None:
            return None

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _dispatch(self, method: str | None, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "omni-memory", "version": "0.1.0"},
            }

        if method == "ping":
            return {}

        if method == "tools/list":
            return {"tools": MCP_TOOL_SCHEMAS}

        if method == "tools/call":
            return self._call_tool(params)

        raise ValueError(f"Unsupported MCP method: {method}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name not in self._handlers:
            raise ValueError(f"Unknown tool: {name}")

        result = self._handlers[name](**arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2),
                }
            ],
            "isError": False,
        }


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]

    return value


def serve_stdio(memory: OmniMemory) -> None:
    StdioMcpServer(memory).serve()
