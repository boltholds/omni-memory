from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.integrations.mcp import build_mcp_handlers
from app.memory import OmniMemory


def build_mcp_app(memory: OmniMemory) -> FastMCP:
    """Build the official MCP SDK server for an OmniMemory instance."""
    server = FastMCP("omni-memory")
    handlers = build_mcp_handlers(memory)

    def call(name: str, **kwargs: Any) -> str:
        result = handlers[name](**kwargs)
        return json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2)

    @server.tool(
        name="omni_memory_write_items",
        description="Save memory items through OmniMemory writeback policies.",
        structured_output=False,
    )
    def write_items(
        items: list[dict[str, Any]],
        source: str = "mcp",
        dry_run: bool = False,
    ) -> str:
        return call("omni_memory_write_items", items=items, source=source, dry_run=dry_run)

    @server.tool(
        name="omni_memory_retrieve",
        description="Retrieve facts, episodes and semantic chunks from OmniMemory.",
        structured_output=False,
    )
    def retrieve(query: str, k_sem: int = 5, k_eps: int = 3) -> str:
        return call("omni_memory_retrieve", query=query, k_sem=k_sem, k_eps=k_eps)

    @server.tool(
        name="omni_memory_ask",
        description="Ask a question using OmniMemory context and the configured LLM.",
        structured_output=False,
    )
    def ask(question: str, lang: str = "en", style: str = "concise") -> str:
        return call("omni_memory_ask", question=question, lang=lang, style=style)

    @server.tool(
        name="omni_memory_context",
        description="Build an explainable OmniMemory context pack for a query.",
        structured_output=False,
    )
    def context(query: str = "") -> str:
        return call("omni_memory_context", query=query)

    @server.tool(
        name="omni_memory_detect_conflicts",
        description="Detect conflicts either for provided facts or for facts retrieved by query.",
        structured_output=False,
    )
    def detect_conflicts(
        query: str | None = None,
        facts: list[dict[str, Any]] | None = None,
    ) -> str:
        return call("omni_memory_detect_conflicts", query=query, facts=facts)

    @server.tool(
        name="omni_memory_write_fact",
        description="Save a single structured fact.",
        structured_output=False,
    )
    def write_fact(
        subject: str,
        predicate: str,
        object: str,
        source: str = "mcp",
        confidence: float = 1.0,
    ) -> str:
        return call(
            "omni_memory_write_fact",
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            confidence=confidence,
        )

    @server.tool(
        name="omni_memory_write_note",
        description="Save a semantic note.",
        structured_output=False,
    )
    def write_note(
        text: str,
        source: str = "mcp",
        meta: dict[str, Any] | None = None,
    ) -> str:
        return call("omni_memory_write_note", text=text, source=source, meta=meta or {})

    @server.tool(
        name="omni_memory_session_ingest_turn",
        description="Append a turn to the in-process session buffer before session distillation.",
        structured_output=False,
    )
    def session_ingest_turn(role: str, content: str) -> str:
        return call("omni_memory_session_ingest_turn", role=role, content=content)

    @server.tool(
        name="omni_memory_session_commit",
        description="Distill buffered session turns into durable memory.",
        structured_output=False,
    )
    def session_commit(
        source: str = "mcp-session",
        dry_run: bool = False,
        min_confidence: float = 0.75,
        clear: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> str:
        return call(
            "omni_memory_session_commit",
            source=source,
            dry_run=dry_run,
            min_confidence=min_confidence,
            clear=clear,
            meta=meta or {},
        )

    @server.tool(
        name="omni_memory_session_clear",
        description="Clear buffered session turns without writing memory.",
        structured_output=False,
    )
    def session_clear() -> str:
        return call("omni_memory_session_clear")

    @server.tool(
        name="omni_memory_stats",
        description="Return lightweight repository counts for the local OmniMemory instance.",
        structured_output=False,
    )
    def stats() -> str:
        return call("omni_memory_stats")

    return server


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
