from __future__ import annotations

from typing import Any, Callable

from app.memory import OmniMemory


MCP_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "omni_memory_write_items",
        "description": "Save memory items through OmniMemory writeback policies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
                "source": {"type": "string", "default": "mcp"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["items"],
        },
    },
    {
        "name": "omni_memory_retrieve",
        "description": "Retrieve facts, episodes and semantic chunks from OmniMemory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k_sem": {"type": "integer", "default": 5},
                "k_eps": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "omni_memory_ask",
        "description": "Ask a question using OmniMemory context and the configured LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "lang": {"type": "string", "default": "en"},
                "style": {"type": "string", "default": "concise"},
            },
            "required": ["question"],
        },
    },
]


def build_mcp_handlers(memory: OmniMemory) -> dict[str, Callable[..., Any]]:
    return {
        "omni_memory_write_items": lambda **kwargs: memory.write_items(
            kwargs["items"],
            source=kwargs.get("source", "mcp"),
            dry_run=kwargs.get("dry_run", False),
        ).model_dump(),
        "omni_memory_retrieve": lambda **kwargs: memory.retrieve(
            kwargs["query"],
            k_sem=kwargs.get("k_sem", 5),
            k_eps=kwargs.get("k_eps", 3),
        ).model_dump(),
        "omni_memory_ask": lambda **kwargs: memory.ask(
            kwargs["question"],
            lang=kwargs.get("lang", "en"),
            style=kwargs.get("style", "concise"),
        ).__dict__,
    }
