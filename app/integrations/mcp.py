from __future__ import annotations

from typing import Any, Callable

from domain.models import Fact
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
    {
        "name": "omni_memory_context",
        "description": "Build an explainable OmniMemory context pack for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": ""},
            },
        },
    },
    {
        "name": "omni_memory_detect_conflicts",
        "description": "Detect conflicts either for provided facts or for facts retrieved by query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "facts": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
    {
        "name": "omni_memory_write_fact",
        "description": "Save a single structured fact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "confidence": {"type": "number", "default": 1.0},
            },
            "required": ["subject", "predicate", "object"],
        },
    },
    {
        "name": "omni_memory_write_note",
        "description": "Save a semantic note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            "required": ["text"],
        },
    },
    {
        "name": "omni_memory_session_ingest_turn",
        "description": "Append a turn to the in-process session buffer before session distillation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["role", "content"],
        },
    },
    {
        "name": "omni_memory_session_commit",
        "description": "Distill buffered session turns into durable memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "default": "mcp-session"},
                "dry_run": {"type": "boolean", "default": False},
                "min_confidence": {"type": "number", "default": 0.75},
                "clear": {"type": "boolean", "default": True},
                "meta": {"type": "object", "default": {}},
            },
        },
    },
    {
        "name": "omni_memory_session_clear",
        "description": "Clear buffered session turns without writing memory.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "omni_memory_stats",
        "description": "Return lightweight repository counts for the local OmniMemory instance.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def build_mcp_handlers(memory: OmniMemory) -> dict[str, Callable[..., Any]]:
    def detect_conflicts(**kwargs):
        raw_facts = kwargs.get("facts")
        if raw_facts is not None:
            facts = [Fact.model_validate(item) for item in raw_facts]
            return memory.consistency.detect_conflicts(facts).model_dump()
        return memory.detect_conflicts(kwargs.get("query")).model_dump()

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
        "omni_memory_context": lambda **kwargs: memory.build_context(
            kwargs.get("query", "")
        ).model_dump(),
        "omni_memory_detect_conflicts": detect_conflicts,
        "omni_memory_write_fact": lambda **kwargs: memory.write_fact(
            kwargs["subject"],
            kwargs["predicate"],
            kwargs["object"],
            source=kwargs.get("source", "mcp"),
            confidence=kwargs.get("confidence", 1.0),
        ).model_dump(),
        "omni_memory_write_note": lambda **kwargs: memory.write_note(
            kwargs["text"],
            source=kwargs.get("source", "mcp"),
            meta=kwargs.get("meta") or {},
        ).model_dump(),
        "omni_memory_session_ingest_turn": lambda **kwargs: _session_ingest_turn(
            memory,
            role=kwargs["role"],
            content=kwargs["content"],
        ),
        "omni_memory_session_commit": lambda **kwargs: memory.commit_session(
            source=kwargs.get("source", "mcp-session"),
            dry_run=kwargs.get("dry_run", False),
            meta=kwargs.get("meta") or {},
            min_confidence=kwargs.get("min_confidence", 0.75),
            clear=kwargs.get("clear", True),
        ).model_dump(),
        "omni_memory_session_clear": lambda **kwargs: _session_clear(memory),
        "omni_memory_stats": lambda **kwargs: _stats(memory),
    }


def _session_ingest_turn(memory: OmniMemory, *, role: str, content: str) -> dict[str, Any]:
    memory.ingest_turn(role, content)
    return {"ok": True, "session_turns": len(memory._session_turns)}


def _session_clear(memory: OmniMemory) -> dict[str, Any]:
    memory.clear_session()
    return {"ok": True, "session_turns": 0}


def _stats(memory: OmniMemory) -> dict[str, Any]:
    def count(repo: Any) -> int | None:
        return repo.count() if hasattr(repo, "count") else None

    return {
        "vector_objects": count(memory.vector_repo),
        "facts": count(memory.graph_repo),
        "episodes": count(memory.episodic_repo),
        "session_turns": len(memory._session_turns),
        "llm_configured": memory.llm is not None,
    }
