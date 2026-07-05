from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from omni_memory.integrations.mcp import build_mcp_handlers
from omni_memory.integrations.mcp_registry import MCP_TOOL_REGISTRY, McpToolDefinition
from omni_memory.memory import OmniMemory


_MISSING = object()


def build_mcp_app(memory: OmniMemory) -> FastMCP:
    """Build the official MCP SDK server for an OmniMemory instance."""
    server = FastMCP("omni-memory")
    handlers = build_mcp_handlers(memory)
    missing = [definition.name for definition in MCP_TOOL_REGISTRY if definition.name not in handlers]
    if missing:
        raise ValueError(f"MCP registry has tools without handlers: {missing}")

    for definition in MCP_TOOL_REGISTRY:
        _register_tool(server, definition, handlers[definition.name])

    return server


def _register_tool(
    server: FastMCP,
    definition: McpToolDefinition,
    handler: Callable[..., Any],
) -> None:
    tool_fn = _build_tool_function(definition, handler)
    server.tool(
        name=definition.name,
        description=definition.description or definition.name.replace("_", " "),
        structured_output=False,
    )(tool_fn)


def _build_tool_function(
    definition: McpToolDefinition,
    handler: Callable[..., Any],
) -> Callable[..., str]:
    properties = _tool_properties(definition)
    required = set(_tool_required(definition))
    defaults = {
        name: _MISSING if name in required else schema.get("default", _MISSING)
        for name, schema in properties.items()
    }

    def tool(**kwargs: Any) -> str:
        normalized = _apply_schema_defaults(kwargs, defaults)
        result = handler(**normalized)
        return json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2)

    tool.__name__ = definition.name
    tool.__qualname__ = definition.name
    tool.__doc__ = definition.description or definition.name.replace("_", " ")
    tool.__signature__ = _signature_from_tool_definition(definition)
    tool.__annotations__ = {
        name: _annotation_from_schema(schema) for name, schema in properties.items()
    } | {"return": str}
    return tool


def _tool_properties(definition: McpToolDefinition) -> dict[str, Any]:
    schema = definition.to_schema().get("inputSchema", {})
    return schema.get("properties") or {}


def _tool_required(definition: McpToolDefinition) -> list[str]:
    schema = definition.to_schema().get("inputSchema", {})
    return list(schema.get("required") or [])


def _signature_from_tool_definition(definition: McpToolDefinition) -> inspect.Signature:
    properties = _tool_properties(definition)
    required = set(_tool_required(definition))
    ordered_names = [name for name in properties if name in required] + [
        name for name in properties if name not in required
    ]

    parameters = []
    for name in ordered_names:
        schema = properties[name]
        default = inspect.Parameter.empty if name in required else schema.get("default", None)
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=_annotation_from_schema(schema),
            )
        )

    return inspect.Signature(parameters, return_annotation=str)


def _annotation_from_schema(schema: dict[str, Any]) -> type[Any] | Any:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


def _apply_schema_defaults(kwargs: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    for name, default in defaults.items():
        if name not in normalized and default is not _MISSING:
            normalized[name] = default
    return normalized


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
    build_mcp_app(memory).run("stdio")
