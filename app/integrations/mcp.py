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
        "name": "omni_memory_list_facts",
        "description": "List stored facts with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "omni_memory_get_fact",
        "description": "Get a stored fact by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"fact_id": {"type": "string"}},
            "required": ["fact_id"],
        },
    },
    {
        "name": "omni_memory_patch_fact",
        "description": "Patch a stored fact in place through fact maintenance strategies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string"},
                "patch": {"type": "object"},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["fact_id", "patch"],
        },
    },
    {
        "name": "omni_memory_retract_fact",
        "description": "Soft-delete a fact by marking it retracted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string"},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "omni_memory_supersede_fact",
        "description": "Create a new current fact and mark an old fact historical.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string"},
                "new_fact": {"type": "object"},
                "reason": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["fact_id", "new_fact"],
        },
    },
    {
        "name": "omni_memory_delete_fact",
        "description": "Delete a fact. Soft delete by default; hard=true removes storage record.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string"},
                "hard": {"type": "boolean", "default": False},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["fact_id"],
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
        "name": "omni_memory_write_decision",
        "description": "Save a project decision/ADR record.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "decision": {"type": "string"},
                "context": {"type": "string", "default": ""},
                "consequences": {"type": "array", "items": {"type": "string"}, "default": []},
                "alternatives": {"type": "array", "items": {"type": "string"}, "default": []},
                "refs": {"type": "object", "default": {}},
                "status": {"type": "string", "default": "accepted"},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            "required": ["title", "decision"],
        },
    },
    {
        "name": "omni_memory_list_decisions",
        "description": "List project decision/ADR records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "omni_memory_get_decision",
        "description": "Get a project decision/ADR record by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"decision_id": {"type": "string"}},
            "required": ["decision_id"],
        },
    },
    {
        "name": "omni_memory_write_experience",
        "description": "Save an agent experience record: goal, action, outcome, lesson and reuse conditions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "lesson": {"type": "string"},
                "context": {"type": "string", "default": ""},
                "decision": {"type": "string", "default": ""},
                "actions": {"type": "array", "items": {"type": "string"}, "default": []},
                "outcome": {"type": "string", "default": ""},
                "evaluation": {"type": "object", "default": {}},
                "reuse_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "avoid_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "confidence": {"type": "number", "default": 0.5},
                "refs": {"type": "object", "default": {}},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            "required": ["goal", "lesson"],
        },
    },
    {
        "name": "omni_memory_list_experiences",
        "description": "List agent experience records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "omni_memory_get_experience",
        "description": "Get an agent experience record by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"experience_id": {"type": "string"}},
            "required": ["experience_id"],
        },
    },
    {
        "name": "omni_memory_search_experiences",
        "description": "Search agent experience records by intent, lesson or reuse condition.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "omni_memory_record_agent_cycle",
        "description": "Record a completed agent cycle as reusable experience.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "plan": {"type": "array", "items": {"type": "string"}, "default": []},
                "decisions": {"type": "array", "items": {"type": "string"}, "default": []},
                "actions": {"type": "array", "items": {"type": "string"}, "default": []},
                "outcome": {"type": "string", "default": ""},
                "tests": {"type": "array", "items": {"type": "string"}, "default": []},
                "files": {"type": "array", "items": {"type": "string"}, "default": []},
                "side_effects": {"type": "array", "items": {"type": "string"}, "default": []},
                "lesson": {"type": "string"},
                "reuse_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "avoid_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "confidence": {"type": "number", "default": 0.8},
                "source": {"type": "string", "default": "mcp-agent-cycle"},
                "meta": {"type": "object", "default": {}},
            },
            "required": ["goal", "lesson"],
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
        "name": "omni_memory_clear",
        "description": "Clear durable OmniMemory stores and/or the in-process session buffer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_vectors": {"type": "boolean", "default": True},
                "include_facts": {"type": "boolean", "default": True},
                "include_episodes": {"type": "boolean", "default": True},
                "include_decisions": {"type": "boolean", "default": True},
                "include_experiences": {"type": "boolean", "default": True},
                "include_session": {"type": "boolean", "default": True},
                "dry_run": {"type": "boolean", "default": False},
            },
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
        "omni_memory_list_facts": lambda **kwargs: memory.maintain_facts(
            {
                "operation": "list",
                "subject": kwargs.get("subject"),
                "predicate": kwargs.get("predicate"),
                "object": kwargs.get("object"),
                "status": kwargs.get("status"),
                "limit": kwargs.get("limit"),
            }
        ).model_dump(mode="json"),
        "omni_memory_get_fact": lambda **kwargs: memory.maintain_facts(
            {"operation": "get", "fact_id": kwargs["fact_id"]}
        ).model_dump(mode="json"),
        "omni_memory_patch_fact": lambda **kwargs: memory.maintain_facts(
            {
                "operation": "patch",
                "fact_id": kwargs["fact_id"],
                "patch": kwargs.get("patch") or {},
                "reason": kwargs.get("reason"),
                "dry_run": kwargs.get("dry_run", False),
            }
        ).model_dump(mode="json"),
        "omni_memory_retract_fact": lambda **kwargs: memory.maintain_facts(
            {
                "operation": "retract",
                "fact_id": kwargs["fact_id"],
                "reason": kwargs.get("reason"),
                "dry_run": kwargs.get("dry_run", False),
            }
        ).model_dump(mode="json"),
        "omni_memory_supersede_fact": lambda **kwargs: memory.maintain_facts(
            {
                "operation": "supersede",
                "fact_id": kwargs["fact_id"],
                "new_fact": kwargs.get("new_fact") or {},
                "reason": kwargs.get("reason"),
                "source": kwargs.get("source", "mcp"),
                "dry_run": kwargs.get("dry_run", False),
            }
        ).model_dump(mode="json"),
        "omni_memory_delete_fact": lambda **kwargs: memory.maintain_facts(
            {
                "operation": "hard_delete" if kwargs.get("hard", False) else "retract",
                "fact_id": kwargs["fact_id"],
                "reason": kwargs.get("reason"),
                "dry_run": kwargs.get("dry_run", False),
            }
        ).model_dump(mode="json"),
        "omni_memory_write_note": lambda **kwargs: memory.write_note(
            kwargs["text"],
            source=kwargs.get("source", "mcp"),
            meta=kwargs.get("meta") or {},
        ).model_dump(),
        "omni_memory_write_decision": lambda **kwargs: memory.write_decision(
            title=kwargs["title"],
            decision=kwargs["decision"],
            context=kwargs.get("context", ""),
            consequences=kwargs.get("consequences") or [],
            alternatives=kwargs.get("alternatives") or [],
            refs=kwargs.get("refs") or {},
            status=kwargs.get("status", "accepted"),
            source=kwargs.get("source", "mcp"),
            meta=kwargs.get("meta") or {},
        ).model_dump(),
        "omni_memory_list_decisions": lambda **kwargs: {
            "decisions": [
                decision.model_dump(mode="json")
                for decision in memory.list_decisions(
                    status=kwargs.get("status"),
                    limit=kwargs.get("limit"),
                )
            ]
        },
        "omni_memory_get_decision": lambda **kwargs: {
            "decision": (
                decision.model_dump(mode="json")
                if (decision := memory.get_decision(kwargs["decision_id"])) is not None
                else None
            )
        },
        "omni_memory_write_experience": lambda **kwargs: memory.record_experience(
            goal=kwargs["goal"],
            lesson=kwargs["lesson"],
            context=kwargs.get("context", ""),
            decision=kwargs.get("decision", ""),
            actions=kwargs.get("actions") or [],
            outcome=kwargs.get("outcome", ""),
            evaluation=kwargs.get("evaluation") or {},
            reuse_when=kwargs.get("reuse_when") or [],
            avoid_when=kwargs.get("avoid_when") or [],
            confidence=kwargs.get("confidence", 0.5),
            refs=kwargs.get("refs") or {},
            source=kwargs.get("source", "mcp"),
            meta=kwargs.get("meta") or {},
        ).model_dump(),
        "omni_memory_list_experiences": lambda **kwargs: {
            "experiences": [
                experience.model_dump(mode="json")
                for experience in memory.list_experiences(limit=kwargs.get("limit"))
            ]
        },
        "omni_memory_get_experience": lambda **kwargs: {
            "experience": (
                experience.model_dump(mode="json")
                if (experience := memory.get_experience(kwargs["experience_id"])) is not None
                else None
            )
        },
        "omni_memory_search_experiences": lambda **kwargs: {
            "experiences": [
                experience.model_dump(mode="json")
                for experience in memory.search_experiences(
                    kwargs["query"],
                    k=kwargs.get("k", 5),
                )
            ]
        },
        "omni_memory_record_agent_cycle": lambda **kwargs: memory.record_agent_cycle(
            {
                "goal": kwargs["goal"],
                "plan": kwargs.get("plan") or [],
                "decisions": kwargs.get("decisions") or [],
                "actions": kwargs.get("actions") or [],
                "outcome": kwargs.get("outcome", ""),
                "tests": kwargs.get("tests") or [],
                "files": kwargs.get("files") or [],
                "side_effects": kwargs.get("side_effects") or [],
                "lesson": kwargs["lesson"],
                "reuse_when": kwargs.get("reuse_when") or [],
                "avoid_when": kwargs.get("avoid_when") or [],
                "confidence": kwargs.get("confidence", 0.8),
                "meta": kwargs.get("meta") or {},
            },
            source=kwargs.get("source", "mcp-agent-cycle"),
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
        "omni_memory_clear": lambda **kwargs: memory.clear(
            include_vectors=kwargs.get("include_vectors", True),
            include_facts=kwargs.get("include_facts", True),
            include_episodes=kwargs.get("include_episodes", True),
            include_decisions=kwargs.get("include_decisions", True),
            include_experiences=kwargs.get("include_experiences", True),
            include_session=kwargs.get("include_session", True),
            dry_run=kwargs.get("dry_run", False),
        ).__dict__,
        "omni_memory_stats": lambda **kwargs: _stats(memory),
    }


def _session_ingest_turn(memory: OmniMemory, *, role: str, content: str) -> dict[str, Any]:
    memory.ingest_turn(role, content)
    return {"ok": True, "session_turns": len(memory._session_turns)}


def _session_clear(memory: OmniMemory) -> dict[str, Any]:
    memory.clear_session()
    return {"ok": True, "session_turns": 0}


def _stats(memory: OmniMemory) -> dict[str, Any]:
    return {
        **memory.repository_stats(),
        "session_turns": len(memory._session_turns),
        "llm_configured": memory.llm is not None,
    }
