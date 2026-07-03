from __future__ import annotations

import time
from typing import Any, Callable

from domain.models import Fact, FailurePatternRecord, Provenance, SkillRecord
from domain.writeback import stable_id
from app.memory import OmniMemory


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


MCP_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "omni_memory_write_items",
        "description": "Save memory items through OmniMemory writeback policies.",
        "inputSchema": _object_schema(
            {
                "items": {"type": "array", "items": {"type": "object"}},
                "source": {"type": "string", "default": "mcp"},
                "dry_run": {"type": "boolean", "default": False},
            },
            ["items"],
        ),
    },
    {
        "name": "omni_memory_retrieve",
        "description": "Retrieve facts, episodes and semantic chunks from OmniMemory.",
        "inputSchema": _object_schema(
            {
                "query": {"type": "string"},
                "k_sem": {"type": "integer", "default": 5},
                "k_eps": {"type": "integer", "default": 3},
                "intent": {"type": "string"},
                "mode": {"type": "string"},
            },
            ["query"],
        ),
    },
    {
        "name": "omni_memory_ask",
        "description": "Ask a question using OmniMemory context and the configured LLM.",
        "inputSchema": _object_schema(
            {
                "question": {"type": "string"},
                "lang": {"type": "string", "default": "en"},
                "style": {"type": "string", "default": "concise"},
                "intent": {"type": "string"},
                "mode": {"type": "string"},
            },
            ["question"],
        ),
    },
    {
        "name": "omni_memory_context",
        "description": "Build an explainable OmniMemory context pack for a query.",
        "inputSchema": _object_schema(
            {
                "query": {"type": "string", "default": ""},
                "intent": {"type": "string"},
                "mode": {"type": "string"},
            }
        ),
    },
    {
        "name": "omni_memory_detect_conflicts",
        "description": "Detect conflicts either for provided facts or for facts retrieved by query.",
        "inputSchema": _object_schema(
            {
                "query": {"type": "string"},
                "facts": {"type": "array", "items": {"type": "object"}},
            }
        ),
    },
    {
        "name": "omni_memory_write_fact",
        "description": "Save a single structured fact.",
        "inputSchema": _object_schema(
            {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "confidence": {"type": "number", "default": 1.0},
            },
            ["subject", "predicate", "object"],
        ),
    },
    {
        "name": "omni_memory_list_facts",
        "description": "List stored facts with optional filters.",
        "inputSchema": _object_schema(
            {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            }
        ),
    },
    {
        "name": "omni_memory_get_fact",
        "description": "Get a stored fact by id.",
        "inputSchema": _object_schema({"fact_id": {"type": "string"}}, ["fact_id"]),
    },
    {
        "name": "omni_memory_patch_fact",
        "description": "Patch a stored fact in place through fact maintenance strategies.",
        "inputSchema": _object_schema(
            {
                "fact_id": {"type": "string"},
                "patch": {"type": "object"},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            ["fact_id", "patch"],
        ),
    },
    {
        "name": "omni_memory_retract_fact",
        "description": "Soft-delete a fact by marking it retracted.",
        "inputSchema": _object_schema(
            {
                "fact_id": {"type": "string"},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            ["fact_id"],
        ),
    },
    {
        "name": "omni_memory_supersede_fact",
        "description": "Create a new current fact and mark an old fact historical.",
        "inputSchema": _object_schema(
            {
                "fact_id": {"type": "string"},
                "new_fact": {"type": "object"},
                "reason": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "dry_run": {"type": "boolean", "default": False},
            },
            ["fact_id", "new_fact"],
        ),
    },
    {
        "name": "omni_memory_delete_fact",
        "description": "Delete a fact. Soft delete by default; hard=true removes storage record.",
        "inputSchema": _object_schema(
            {
                "fact_id": {"type": "string"},
                "hard": {"type": "boolean", "default": False},
                "reason": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            ["fact_id"],
        ),
    },
    {
        "name": "omni_memory_write_note",
        "description": "Save a semantic note.",
        "inputSchema": _object_schema(
            {
                "text": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            ["text"],
        ),
    },
    {
        "name": "omni_memory_write_decision",
        "description": "Save a project decision/ADR record.",
        "inputSchema": _object_schema(
            {
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
            ["title", "decision"],
        ),
    },
    {
        "name": "omni_memory_list_decisions",
        "description": "List project decision/ADR records.",
        "inputSchema": _object_schema({"status": {"type": "string"}, "limit": {"type": "integer"}}),
    },
    {
        "name": "omni_memory_get_decision",
        "description": "Get a project decision/ADR record by id.",
        "inputSchema": _object_schema({"decision_id": {"type": "string"}}, ["decision_id"]),
    },
    {
        "name": "omni_memory_write_experience",
        "description": "Save an agent experience record: goal, action, outcome, lesson and reuse conditions.",
        "inputSchema": _object_schema(
            {
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
            ["goal", "lesson"],
        ),
    },
    {
        "name": "omni_memory_list_experiences",
        "description": "List agent experience records.",
        "inputSchema": _object_schema({"limit": {"type": "integer"}}),
    },
    {
        "name": "omni_memory_get_experience",
        "description": "Get an agent experience record by id.",
        "inputSchema": _object_schema({"experience_id": {"type": "string"}}, ["experience_id"]),
    },
    {
        "name": "omni_memory_search_experiences",
        "description": "Search agent experience records by intent, lesson or reuse condition.",
        "inputSchema": _object_schema({"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, ["query"]),
    },
    {
        "name": "omni_memory_write_skill",
        "description": "Save a reusable skill record promoted from repeated experience.",
        "inputSchema": _object_schema(
            {
                "name": {"type": "string"},
                "problem": {"type": "string", "default": ""},
                "procedure": {"type": "array", "items": {"type": "string"}, "default": []},
                "reuse_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "avoid_when": {"type": "array", "items": {"type": "string"}, "default": []},
                "evidence_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                "confidence": {"type": "number", "default": 0.5},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            ["name"],
        ),
    },
    {
        "name": "omni_memory_list_skills",
        "description": "List reusable skill records.",
        "inputSchema": _object_schema({"limit": {"type": "integer"}}),
    },
    {
        "name": "omni_memory_get_skill",
        "description": "Get a reusable skill record by id.",
        "inputSchema": _object_schema({"skill_id": {"type": "string"}}, ["skill_id"]),
    },
    {
        "name": "omni_memory_search_skills",
        "description": "Search reusable skill records.",
        "inputSchema": _object_schema({"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, ["query"]),
    },
    {
        "name": "omni_memory_write_failure_pattern",
        "description": "Save a reusable failure pattern with symptom, cause, fix and detection hints.",
        "inputSchema": _object_schema(
            {
                "symptom": {"type": "string"},
                "root_cause": {"type": "string", "default": ""},
                "fix": {"type": "string", "default": ""},
                "detection": {"type": "string", "default": ""},
                "evidence_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                "confidence": {"type": "number", "default": 0.5},
                "source": {"type": "string", "default": "mcp"},
                "meta": {"type": "object", "default": {}},
            },
            ["symptom"],
        ),
    },
    {
        "name": "omni_memory_list_failure_patterns",
        "description": "List reusable failure pattern records.",
        "inputSchema": _object_schema({"limit": {"type": "integer"}}),
    },
    {
        "name": "omni_memory_get_failure_pattern",
        "description": "Get a reusable failure pattern record by id.",
        "inputSchema": _object_schema({"pattern_id": {"type": "string"}}, ["pattern_id"]),
    },
    {
        "name": "omni_memory_search_failure_patterns",
        "description": "Search reusable failure pattern records.",
        "inputSchema": _object_schema({"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, ["query"]),
    },
    {
        "name": "omni_memory_record_agent_cycle",
        "description": "Record a completed agent cycle as reusable experience.",
        "inputSchema": _object_schema(
            {
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
            ["goal", "lesson"],
        ),
    },
    {
        "name": "omni_memory_session_ingest_turn",
        "description": "Append a turn to the in-process session buffer before session distillation.",
        "inputSchema": _object_schema({"role": {"type": "string"}, "content": {"type": "string"}}, ["role", "content"]),
    },
    {
        "name": "omni_memory_session_commit",
        "description": "Distill buffered session turns into durable memory.",
        "inputSchema": _object_schema(
            {
                "source": {"type": "string", "default": "mcp-session"},
                "dry_run": {"type": "boolean", "default": False},
                "min_confidence": {"type": "number", "default": 0.75},
                "clear": {"type": "boolean", "default": True},
                "meta": {"type": "object", "default": {}},
            }
        ),
    },
    {"name": "omni_memory_session_clear", "description": "Clear buffered session turns without writing memory.", "inputSchema": _object_schema({})},
    {
        "name": "omni_memory_clear",
        "description": "Clear durable OmniMemory stores and/or the in-process session buffer.",
        "inputSchema": _object_schema(
            {
                "include_vectors": {"type": "boolean", "default": True},
                "include_facts": {"type": "boolean", "default": True},
                "include_episodes": {"type": "boolean", "default": True},
                "include_decisions": {"type": "boolean", "default": True},
                "include_experiences": {"type": "boolean", "default": True},
                "include_session": {"type": "boolean", "default": True},
                "dry_run": {"type": "boolean", "default": False},
            }
        ),
    },
    {"name": "omni_memory_stats", "description": "Return lightweight repository counts for the local OmniMemory instance.", "inputSchema": _object_schema({})},
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
            intent=kwargs.get("intent"),
            mode=kwargs.get("mode"),
        ).model_dump(),
        "omni_memory_ask": lambda **kwargs: memory.ask(
            kwargs["question"],
            lang=kwargs.get("lang", "en"),
            style=kwargs.get("style", "concise"),
            intent=kwargs.get("intent"),
            mode=kwargs.get("mode"),
        ).__dict__,
        "omni_memory_context": lambda **kwargs: memory.build_context(
            kwargs.get("query", ""),
            intent=kwargs.get("intent"),
            mode=kwargs.get("mode"),
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
            {"operation": "patch", "fact_id": kwargs["fact_id"], "patch": kwargs.get("patch") or {}, "reason": kwargs.get("reason"), "dry_run": kwargs.get("dry_run", False)}
        ).model_dump(mode="json"),
        "omni_memory_retract_fact": lambda **kwargs: memory.maintain_facts(
            {"operation": "retract", "fact_id": kwargs["fact_id"], "reason": kwargs.get("reason"), "dry_run": kwargs.get("dry_run", False)}
        ).model_dump(mode="json"),
        "omni_memory_supersede_fact": lambda **kwargs: memory.maintain_facts(
            {"operation": "supersede", "fact_id": kwargs["fact_id"], "new_fact": kwargs.get("new_fact") or {}, "reason": kwargs.get("reason"), "source": kwargs.get("source", "mcp"), "dry_run": kwargs.get("dry_run", False)}
        ).model_dump(mode="json"),
        "omni_memory_delete_fact": lambda **kwargs: memory.maintain_facts(
            {"operation": "hard_delete" if kwargs.get("hard", False) else "retract", "fact_id": kwargs["fact_id"], "reason": kwargs.get("reason"), "dry_run": kwargs.get("dry_run", False)}
        ).model_dump(mode="json"),
        "omni_memory_write_note": lambda **kwargs: memory.write_note(kwargs["text"], source=kwargs.get("source", "mcp"), meta=kwargs.get("meta") or {}).model_dump(),
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
        "omni_memory_list_decisions": lambda **kwargs: {"decisions": [decision.model_dump(mode="json") for decision in memory.list_decisions(status=kwargs.get("status"), limit=kwargs.get("limit"))]},
        "omni_memory_get_decision": lambda **kwargs: {"decision": (decision.model_dump(mode="json") if (decision := memory.get_decision(kwargs["decision_id"])) is not None else None)},
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
        "omni_memory_list_experiences": lambda **kwargs: {"experiences": [experience.model_dump(mode="json") for experience in memory.list_experiences(limit=kwargs.get("limit"))]},
        "omni_memory_get_experience": lambda **kwargs: {"experience": (experience.model_dump(mode="json") if (experience := memory.get_experience(kwargs["experience_id"])) is not None else None)},
        "omni_memory_search_experiences": lambda **kwargs: {"experiences": [experience.model_dump(mode="json") for experience in memory.search_experiences(kwargs["query"], k=kwargs.get("k", 5))]},
        "omni_memory_write_skill": lambda **kwargs: _write_skill(memory, **kwargs),
        "omni_memory_list_skills": lambda **kwargs: {"skills": [skill.model_dump(mode="json") for skill in memory.repositories.skill.list_skills(limit=kwargs.get("limit"))]},
        "omni_memory_get_skill": lambda **kwargs: {"skill": (skill.model_dump(mode="json") if (skill := memory.repositories.skill.get_skill(kwargs["skill_id"])) is not None else None)},
        "omni_memory_search_skills": lambda **kwargs: {"skills": [skill.model_dump(mode="json") for skill in memory.repositories.skill.search(kwargs["query"], k=kwargs.get("k", 5))]},
        "omni_memory_write_failure_pattern": lambda **kwargs: _write_failure_pattern(memory, **kwargs),
        "omni_memory_list_failure_patterns": lambda **kwargs: {"failure_patterns": [pattern.model_dump(mode="json") for pattern in memory.repositories.failure_pattern.list_failure_patterns(limit=kwargs.get("limit"))]},
        "omni_memory_get_failure_pattern": lambda **kwargs: {"failure_pattern": (pattern.model_dump(mode="json") if (pattern := memory.repositories.failure_pattern.get_failure_pattern(kwargs["pattern_id"])) is not None else None)},
        "omni_memory_search_failure_patterns": lambda **kwargs: {"failure_patterns": [pattern.model_dump(mode="json") for pattern in memory.repositories.failure_pattern.search(kwargs["query"], k=kwargs.get("k", 5))]},
        "omni_memory_record_agent_cycle": lambda **kwargs: memory.record_agent_cycle(
            {"goal": kwargs["goal"], "plan": kwargs.get("plan") or [], "decisions": kwargs.get("decisions") or [], "actions": kwargs.get("actions") or [], "outcome": kwargs.get("outcome", ""), "tests": kwargs.get("tests") or [], "files": kwargs.get("files") or [], "side_effects": kwargs.get("side_effects") or [], "lesson": kwargs["lesson"], "reuse_when": kwargs.get("reuse_when") or [], "avoid_when": kwargs.get("avoid_when") or [], "confidence": kwargs.get("confidence", 0.8), "meta": kwargs.get("meta") or {}},
            source=kwargs.get("source", "mcp-agent-cycle"),
        ).model_dump(),
        "omni_memory_session_ingest_turn": lambda **kwargs: _session_ingest_turn(memory, role=kwargs["role"], content=kwargs["content"]),
        "omni_memory_session_commit": lambda **kwargs: memory.commit_session(source=kwargs.get("source", "mcp-session"), dry_run=kwargs.get("dry_run", False), meta=kwargs.get("meta") or {}, min_confidence=kwargs.get("min_confidence", 0.75), clear=kwargs.get("clear", True)).model_dump(),
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


def _write_skill(memory: OmniMemory, **kwargs) -> dict[str, Any]:
    payload = {
        "name": kwargs["name"],
        "problem": kwargs.get("problem", ""),
        "procedure": kwargs.get("procedure") or [],
        "reuse_when": kwargs.get("reuse_when") or [],
        "avoid_when": kwargs.get("avoid_when") or [],
        "evidence_ids": kwargs.get("evidence_ids") or [],
        "confidence": kwargs.get("confidence", 0.5),
    }
    skill = SkillRecord(
        id=stable_id("skill", payload),
        refs=kwargs.get("refs") or {},
        provenance=Provenance(source=kwargs.get("source", "mcp"), time=time.time()),
        meta=kwargs.get("meta") or {},
        **payload,
    )
    memory.repositories.skill.save_skill(skill)
    return {"saved": 1, "skill": skill.model_dump(mode="json")}


def _write_failure_pattern(memory: OmniMemory, **kwargs) -> dict[str, Any]:
    payload = {
        "symptom": kwargs["symptom"],
        "root_cause": kwargs.get("root_cause", ""),
        "fix": kwargs.get("fix", ""),
        "detection": kwargs.get("detection", ""),
        "evidence_ids": kwargs.get("evidence_ids") or [],
        "confidence": kwargs.get("confidence", 0.5),
    }
    pattern = FailurePatternRecord(
        id=stable_id("failure_pattern", payload),
        refs=kwargs.get("refs") or {},
        provenance=Provenance(source=kwargs.get("source", "mcp"), time=time.time()),
        meta=kwargs.get("meta") or {},
        **payload,
    )
    memory.repositories.failure_pattern.save_failure_pattern(pattern)
    return {"saved": 1, "failure_pattern": pattern.model_dump(mode="json")}


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
