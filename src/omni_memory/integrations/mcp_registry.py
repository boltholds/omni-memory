from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_memory.integrations.fact_mining_mcp import FACT_MINING_TOOL_SCHEMA


MCP_PROFILE_AGENT_CORE = "agent_core"
MCP_PROFILE_MAINTENANCE = "maintenance"
MCP_PROFILES = (MCP_PROFILE_AGENT_CORE, MCP_PROFILE_MAINTENANCE)


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    properties: dict[str, Any] | None = None
    required: list[str] | None = None
    description: str | None = None
    schema_override: dict[str, Any] | None = None
    profile: str = MCP_PROFILE_MAINTENANCE

    def to_schema(self) -> dict[str, Any]:
        if self.schema_override is not None:
            return self.schema_override
        return {
            "name": self.name,
            "description": self.description or self.name.replace("_", " "),
            "inputSchema": _object_schema(self.properties or {}, self.required),
        }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_NUMBER = {"type": "number"}
_BOOLEAN = {"type": "boolean"}


def mcp_tool_schemas() -> list[dict[str, Any]]:
    return [definition.to_schema() for definition in MCP_TOOL_REGISTRY]


def mcp_tool_definitions_for_profile(profile: str) -> list[McpToolDefinition]:
    if profile not in MCP_PROFILES:
        raise ValueError(f"Unknown MCP tool profile: {profile}")
    return [definition for definition in MCP_TOOL_REGISTRY if definition.profile == profile]


def mcp_tool_names_for_profile(profile: str) -> list[str]:
    return [definition.name for definition in mcp_tool_definitions_for_profile(profile)]


def mcp_tool_profiles() -> dict[str, list[str]]:
    return {profile: mcp_tool_names_for_profile(profile) for profile in MCP_PROFILES}


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _array(item_type: str = "string") -> dict[str, Any]:
    return {"type": "array", "items": {"type": item_type}, "default": []}


def _object(default: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "object", "default": default or {}}


def _scope_props() -> dict[str, Any]:
    return {
        "domain_ids": {"type": "array", "items": {"type": "string"}},
        "domains": {"type": "array", "items": {"type": "string"}},
        "environments": {"type": "array", "items": {"type": "string"}},
        "environment": _STRING,
        "durabilities": {"type": "array", "items": {"type": "string"}},
        "durability": _STRING,
        "memory_types": {"type": "array", "items": {"type": "string"}},
        "types": {"type": "array", "items": {"type": "string"}},
        "include_ephemeral": {"type": "boolean", "default": True},
        "strict_domains": {"type": "boolean", "default": False},
        "expand_domains": {"type": "boolean", "default": True},
    }


def _scope_schema() -> dict[str, Any]:
    return _object_schema(_scope_props())


def _write_items_props() -> dict[str, Any]:
    return {
        "items": {"type": "array", "items": {"type": "object"}, "default": []},
        "source": {"type": "string", "default": "mcp"},
        "dry_run": {"type": "boolean", "default": False},
    }


def _fact_props() -> dict[str, Any]:
    return {
        "subject": _STRING,
        "predicate": _STRING,
        "object": _STRING,
        "source": {"type": "string", "default": "mcp"},
        "confidence": {"type": "number", "default": 1.0},
    }


def _list_fact_props() -> dict[str, Any]:
    return {
        "subject": _STRING,
        "predicate": _STRING,
        "object": _STRING,
        "status": _STRING,
        "limit": _INTEGER,
    }


def _patch_fact_props() -> dict[str, Any]:
    return {
        "fact_id": _STRING,
        "patch": _object(),
        "reason": _STRING,
        "dry_run": {"type": "boolean", "default": False},
    }


def _supersede_fact_props() -> dict[str, Any]:
    return {
        "fact_id": _STRING,
        "new_fact": _object(),
        "reason": _STRING,
        "source": {"type": "string", "default": "mcp"},
        "dry_run": {"type": "boolean", "default": False},
    }


def _delete_fact_props() -> dict[str, Any]:
    return {
        "fact_id": _STRING,
        "hard": {"type": "boolean", "default": False},
        "reason": _STRING,
        "dry_run": {"type": "boolean", "default": False},
    }


def _decision_props() -> dict[str, Any]:
    return {
        "title": _STRING,
        "decision": _STRING,
        "context": {"type": "string", "default": ""},
        "consequences": _array(),
        "alternatives": _array(),
        "refs": _object(),
        "status": {"type": "string", "default": "accepted"},
        "source": {"type": "string", "default": "mcp"},
        "meta": _object(),
    }


def _experience_props() -> dict[str, Any]:
    return {
        "goal": _STRING,
        "lesson": _STRING,
        "context": {"type": "string", "default": ""},
        "decision": {"type": "string", "default": ""},
        "actions": _array(),
        "outcome": {"type": "string", "default": ""},
        "evaluation": _object(),
        "reuse_when": _array(),
        "avoid_when": _array(),
        "confidence": {"type": "number", "default": 0.5},
        "refs": _object(),
        "source": {"type": "string", "default": "mcp"},
        "meta": _object(),
    }


def _search_props() -> dict[str, Any]:
    return {"query": _STRING, "k": {"type": "integer", "default": 5}}


def _skill_props() -> dict[str, Any]:
    return {
        "name": _STRING,
        "problem": {"type": "string", "default": ""},
        "procedure": _array(),
        "reuse_when": _array(),
        "avoid_when": _array(),
        "evidence_ids": _array(),
        "confidence": {"type": "number", "default": 0.5},
        "refs": _object(),
        "source": {"type": "string", "default": "mcp"},
        "meta": _object(),
    }


def _failure_pattern_props() -> dict[str, Any]:
    return {
        "symptom": _STRING,
        "root_cause": {"type": "string", "default": ""},
        "fix": {"type": "string", "default": ""},
        "detection": {"type": "string", "default": ""},
        "evidence_ids": _array(),
        "confidence": {"type": "number", "default": 0.5},
        "refs": _object(),
        "source": {"type": "string", "default": "mcp"},
        "meta": _object(),
    }


def _agent_cycle_props() -> dict[str, Any]:
    return {
        "goal": _STRING,
        "lesson": _STRING,
        "plan": _array(),
        "decisions": _array(),
        "actions": _array(),
        "outcome": {"type": "string", "default": ""},
        "tests": _array(),
        "files": _array(),
        "side_effects": _array(),
        "reuse_when": _array(),
        "avoid_when": _array(),
        "confidence": {"type": "number", "default": 0.8},
        "source": {"type": "string", "default": "mcp-agent-cycle"},
        "meta": _object(),
    }


def _development_cycle_props() -> dict[str, Any]:
    return {
        "goal": _STRING,
        "summary": {"type": "string", "default": ""},
        "changed_files": _array(),
        "commands_run": _array(),
        "tests": _array(),
        "decisions": _array(),
        "outcome": {"type": "string", "default": ""},
        "lesson": {"type": "string", "default": ""},
        "reuse_when": _array(),
        "avoid_when": _array(),
        "side_effects": _array(),
        "confidence": {"type": "number", "default": 0.8},
        "source": {"type": "string", "default": "mcp-development-cycle"},
        "meta": _object(),
    }


def _finish_task_props() -> dict[str, Any]:
    return {
        **_development_cycle_props(),
        "source": {"type": "string", "default": "mcp-development-workflow"},
        "session_turns": {"type": "array", "items": {"type": "object"}, "default": []},
        "run_distiller": {"type": "boolean", "default": True},
        "distill_dry_run": {"type": "boolean", "default": True},
        "min_confidence": {"type": "number", "default": 0.75},
        "clear_session": {"type": "boolean", "default": False},
    }


def _ops_cycle_props() -> dict[str, Any]:
    return {
        "goal": _STRING,
        "service": _STRING,
        "alert_id": _STRING,
        "symptoms": _array(),
        "actions": _array(),
        "outcome": {"type": "string", "default": ""},
        "metrics_before": _object(),
        "metrics_after": _object(),
        "lesson": {"type": "string", "default": ""},
        "reuse_when": _array(),
        "avoid_when": _array(),
        "affected_resources": _array(),
        "confidence": {"type": "number", "default": 0.8},
        "source": {"type": "string", "default": "mcp-ops-cycle"},
        "meta": _object(),
    }


def _review_item_props() -> dict[str, Any]:
    return {
        "kind": _STRING,
        "title": _STRING,
        "payload": _object(),
        "confidence": {"type": "number", "default": 0.5},
        "reason": {"type": "string", "default": ""},
        "source": {"type": "string", "default": "mcp-review"},
        "meta": _object(),
    }


def _review_list_props() -> dict[str, Any]:
    return {
        "status": _STRING,
        "kind": _STRING,
        "limit": _INTEGER,
    }


def _review_action_props() -> dict[str, Any]:
    return {
        "item_id": _STRING,
        "reviewer": {"type": "string", "default": "mcp"},
        "note": {"type": "string", "default": ""},
    }


def _session_commit_props() -> dict[str, Any]:
    return {
        "source": {"type": "string", "default": "mcp-session"},
        "dry_run": {"type": "boolean", "default": False},
        "min_confidence": {"type": "number", "default": 0.75},
        "clear": {"type": "boolean", "default": True},
        "meta": _object(),
    }


def _clear_props() -> dict[str, Any]:
    return {
        "include_vectors": {"type": "boolean", "default": True},
        "include_facts": {"type": "boolean", "default": True},
        "include_episodes": {"type": "boolean", "default": True},
        "include_decisions": {"type": "boolean", "default": True},
        "include_experiences": {"type": "boolean", "default": True},
        "include_skills": {"type": "boolean", "default": True},
        "include_failure_patterns": {"type": "boolean", "default": True},
        "include_review_items": {"type": "boolean", "default": True},
        "include_session": {"type": "boolean", "default": True},
        "dry_run": {"type": "boolean", "default": False},
    }


MCP_TOOL_REGISTRY: list[McpToolDefinition] = [
    McpToolDefinition("omni_memory_write_items", _write_items_props(), ["items"], "Save memory items through OmniMemory writeback policies."),
    McpToolDefinition(
        "omni_memory_retrieve",
        {"query": _STRING, "k_sem": {"type": "integer", "default": 5}, "k_eps": {"type": "integer", "default": 3}, "intent": _STRING, "mode": _STRING, "scope": _scope_schema()},
        ["query"],
        "Retrieve facts, episodes and semantic chunks from OmniMemory.",
        profile=MCP_PROFILE_AGENT_CORE,
    ),
    McpToolDefinition(
        "omni_memory_ask",
        {"question": _STRING, "lang": {"type": "string", "default": "en"}, "style": {"type": "string", "default": "concise"}, "intent": _STRING, "mode": _STRING, "scope": _scope_schema()},
        ["question"],
        "Ask a question using OmniMemory context and the configured LLM.",
    ),
    McpToolDefinition("omni_memory_context", {"query": {"type": "string", "default": ""}, "intent": _STRING, "mode": _STRING, "scope": _scope_schema()}, description="Build an explainable OmniMemory context pack for a query.", profile=MCP_PROFILE_AGENT_CORE),
    McpToolDefinition("omni_memory_detect_conflicts", {"query": _STRING, "facts": {"type": "array", "items": {"type": "object"}}, "scope": _scope_schema()}, description="Detect conflicts either for provided facts or for facts retrieved by query."),
    McpToolDefinition("omni_memory_mine_facts", schema_override=FACT_MINING_TOOL_SCHEMA),
    McpToolDefinition("omni_memory_write_fact", _fact_props(), ["subject", "predicate", "object"], "Save a single structured fact."),
    McpToolDefinition("omni_memory_list_facts", _list_fact_props(), description="List stored facts with optional filters."),
    McpToolDefinition("omni_memory_get_fact", {"fact_id": _STRING}, ["fact_id"], "Get a stored fact by id."),
    McpToolDefinition("omni_memory_patch_fact", _patch_fact_props(), ["fact_id"], "Patch a stored fact in place through fact maintenance strategies."),
    McpToolDefinition("omni_memory_retract_fact", {"fact_id": _STRING, "reason": _STRING, "dry_run": {"type": "boolean", "default": False}}, ["fact_id"], "Soft-delete a fact by marking it retracted."),
    McpToolDefinition("omni_memory_supersede_fact", _supersede_fact_props(), ["fact_id", "new_fact"], "Create a new current fact and mark an old fact historical."),
    McpToolDefinition("omni_memory_delete_fact", _delete_fact_props(), ["fact_id"], "Delete a fact. Soft delete by default; hard=true removes storage record."),
    McpToolDefinition("omni_memory_write_note", {"text": _STRING, "source": {"type": "string", "default": "mcp"}, "meta": _object()}, ["text"], "Save a semantic note."),
    McpToolDefinition("omni_memory_write_decision", _decision_props(), ["title", "decision"], "Save a project decision/ADR record."),
    McpToolDefinition("omni_memory_list_decisions", {"status": _STRING, "limit": _INTEGER}, description="List project decision/ADR records."),
    McpToolDefinition("omni_memory_get_decision", {"decision_id": _STRING}, ["decision_id"], "Get a project decision/ADR record by id."),
    McpToolDefinition("omni_memory_write_experience", _experience_props(), ["goal", "lesson"], "Save an agent experience record: goal, action, outcome, lesson and reuse conditions."),
    McpToolDefinition("omni_memory_list_experiences", {"limit": _INTEGER}, description="List agent experience records."),
    McpToolDefinition("omni_memory_get_experience", {"experience_id": _STRING}, ["experience_id"], "Get an agent experience record by id."),
    McpToolDefinition("omni_memory_search_experiences", _search_props(), ["query"], "Search agent experience records by intent, lesson or reuse condition.", profile=MCP_PROFILE_AGENT_CORE),
    McpToolDefinition("omni_memory_write_skill", _skill_props(), ["name"], "Save a reusable skill record promoted from repeated experience."),
    McpToolDefinition("omni_memory_list_skills", {"limit": _INTEGER}, description="List reusable skill records."),
    McpToolDefinition("omni_memory_get_skill", {"skill_id": _STRING}, ["skill_id"], "Get a reusable skill record by id."),
    McpToolDefinition("omni_memory_search_skills", _search_props(), ["query"], "Search reusable skill records."),
    McpToolDefinition("omni_memory_write_failure_pattern", _failure_pattern_props(), ["symptom"], "Save a reusable failure pattern with symptom, cause, fix and detection hints."),
    McpToolDefinition("omni_memory_list_failure_patterns", {"limit": _INTEGER}, description="List reusable failure pattern records."),
    McpToolDefinition("omni_memory_get_failure_pattern", {"pattern_id": _STRING}, ["pattern_id"], "Get a reusable failure pattern record by id."),
    McpToolDefinition("omni_memory_search_failure_patterns", _search_props(), ["query"], "Search reusable failure pattern records.", profile=MCP_PROFILE_AGENT_CORE),
    McpToolDefinition("omni_memory_consolidate_experiences", {"dry_run": {"type": "boolean", "default": True}, "min_confidence": {"type": "number", "default": 0.85}}, description="Consolidate repeated high-confidence experiences into skill and failure-pattern proposals. Dry-run by default."),
    McpToolDefinition("omni_memory_record_agent_cycle", _agent_cycle_props(), ["goal", "lesson"], "Record a completed agent cycle as reusable experience."),
    McpToolDefinition("omni_memory_draft_development_cycle", _development_cycle_props(), ["goal"], "Draft a development cycle as an agent-cycle record without writing memory."),
    McpToolDefinition("omni_memory_record_development_cycle", _development_cycle_props(), ["goal", "lesson"], "Record a development cycle as reusable experience."),
    McpToolDefinition("omni_memory_finish_development_task", _finish_task_props(), ["goal", "lesson"], "Finish a development task: record reusable experience and return distillation review candidates.", profile=MCP_PROFILE_AGENT_CORE),
    McpToolDefinition("omni_memory_draft_ops_cycle", _ops_cycle_props(), ["goal", "service"], "Draft an operations/incident cycle without writing memory."),
    McpToolDefinition("omni_memory_record_ops_cycle", _ops_cycle_props(), ["goal", "service", "lesson"], "Record an operations/incident cycle as reusable experience."),
    McpToolDefinition("omni_memory_submit_review_item", _review_item_props(), ["kind", "title", "payload"], "Submit a cognitive memory proposal to the review queue."),
    McpToolDefinition("omni_memory_list_review_items", _review_list_props(), description="List cognitive memory proposals waiting for review."),
    McpToolDefinition("omni_memory_get_review_item", {"item_id": _STRING}, ["item_id"], "Get a cognitive memory proposal by review item id."),
    McpToolDefinition("omni_memory_accept_review_item", _review_action_props(), ["item_id"], "Accept a cognitive memory proposal and apply it."),
    McpToolDefinition("omni_memory_reject_review_item", _review_action_props(), ["item_id"], "Reject a cognitive memory proposal without applying it."),
    McpToolDefinition("omni_memory_supersede_review_item", {**_review_action_props(), "replacement": _object()}, ["item_id", "replacement"], "Supersede a cognitive memory proposal with a replacement proposal."),
    McpToolDefinition("omni_memory_session_ingest_turn", {"role": _STRING, "content": _STRING}, ["role", "content"], "Append a turn to the in-process session buffer before session distillation."),
    McpToolDefinition("omni_memory_session_commit", _session_commit_props(), description="Distill buffered session turns into durable memory."),
    McpToolDefinition("omni_memory_session_clear", description="Clear buffered session turns without writing memory."),
    McpToolDefinition("omni_memory_clear", _clear_props(), description="Clear durable OmniMemory stores and/or the in-process session buffer."),
    McpToolDefinition("omni_memory_stats", description="Return lightweight repository counts for the local OmniMemory instance."),
]
