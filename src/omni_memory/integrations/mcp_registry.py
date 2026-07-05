from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_memory.integrations.fact_mining_mcp import FACT_MINING_TOOL_SCHEMA


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    properties: dict[str, Any] | None = None
    required: list[str] | None = None
    description: str | None = None
    schema_override: dict[str, Any] | None = None

    def to_schema(self) -> dict[str, Any]:
        if self.schema_override is not None:
            return self.schema_override
        return {
            "name": self.name,
            "description": self.description or self.name.replace("_", " "),
            "inputSchema": _object_schema(self.properties or {}, self.required),
        }


def mcp_tool_schemas() -> list[dict[str, Any]]:
    return [definition.to_schema() for definition in MCP_TOOL_REGISTRY]


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
        "environment": {"type": "string"},
        "durabilities": {"type": "array", "items": {"type": "string"}},
        "durability": {"type": "string"},
        "memory_types": {"type": "array", "items": {"type": "string"}},
        "types": {"type": "array", "items": {"type": "string"}},
        "include_ephemeral": {"type": "boolean", "default": True},
        "strict_domains": {"type": "boolean", "default": False},
        "expand_domains": {"type": "boolean", "default": True},
    }


def _scope_schema() -> dict[str, Any]:
    return _object_schema(_scope_props())


def _fact_props() -> dict[str, Any]:
    return {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
        "source": {"type": "string", "default": "mcp"},
        "confidence": {"type": "number", "default": 1.0},
    }


def _skill_props() -> dict[str, Any]:
    return {
        "name": {"type": "string"},
        "problem": {"type": "string", "default": ""},
        "procedure": _array(),
        "reuse_when": _array(),
        "avoid_when": _array(),
        "evidence_ids": _array(),
        "confidence": {"type": "number", "default": 0.5},
        "source": {"type": "string", "default": "mcp"},
        "meta": _object(),
    }


def _session_commit_props() -> dict[str, Any]:
    return {
        "source": {"type": "string", "default": "mcp-session"},
        "dry_run": {"type": "boolean", "default": False},
        "min_confidence": {"type": "number", "default": 0.75},
        "clear": {"type": "boolean", "default": True},
        "meta": _object(),
    }


def _development_cycle_props() -> dict[str, Any]:
    return {
        "goal": {"type": "string"},
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
        "goal": {"type": "string"},
        "service": {"type": "string"},
        "alert_id": {"type": "string"},
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
        "kind": {"type": "string"},
        "title": {"type": "string"},
        "payload": {"type": "object"},
        "confidence": {"type": "number", "default": 0.5},
        "reason": {"type": "string", "default": ""},
        "source": {"type": "string", "default": "mcp-review"},
        "meta": _object(),
    }


def _review_list_props() -> dict[str, Any]:
    return {
        "status": {"type": "string"},
        "kind": {"type": "string"},
        "limit": {"type": "integer"},
    }


def _review_action_props() -> dict[str, Any]:
    return {
        "item_id": {"type": "string"},
        "reviewer": {"type": "string", "default": "mcp"},
        "note": {"type": "string", "default": ""},
    }


MCP_TOOL_REGISTRY: list[McpToolDefinition] = [
    McpToolDefinition("omni_memory_write_items"),
    McpToolDefinition(
        "omni_memory_retrieve",
        {"query": {"type": "string"}, "k_sem": {"type": "integer", "default": 5}, "k_eps": {"type": "integer", "default": 3}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()},
        ["query"],
    ),
    McpToolDefinition(
        "omni_memory_ask",
        {"question": {"type": "string"}, "lang": {"type": "string", "default": "en"}, "style": {"type": "string", "default": "concise"}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()},
        ["question"],
    ),
    McpToolDefinition("omni_memory_context", {"query": {"type": "string", "default": ""}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()}),
    McpToolDefinition("omni_memory_detect_conflicts", {"query": {"type": "string"}, "facts": {"type": "array", "items": {"type": "object"}}, "scope": _scope_schema()}),
    McpToolDefinition("omni_memory_mine_facts", schema_override=FACT_MINING_TOOL_SCHEMA),
    McpToolDefinition("omni_memory_write_fact", _fact_props(), ["subject", "predicate", "object"]),
    McpToolDefinition("omni_memory_list_facts"),
    McpToolDefinition("omni_memory_get_fact"),
    McpToolDefinition("omni_memory_patch_fact"),
    McpToolDefinition("omni_memory_retract_fact"),
    McpToolDefinition("omni_memory_supersede_fact"),
    McpToolDefinition("omni_memory_delete_fact"),
    McpToolDefinition("omni_memory_write_note"),
    McpToolDefinition("omni_memory_write_decision"),
    McpToolDefinition("omni_memory_list_decisions"),
    McpToolDefinition("omni_memory_get_decision"),
    McpToolDefinition("omni_memory_write_experience"),
    McpToolDefinition("omni_memory_list_experiences"),
    McpToolDefinition("omni_memory_get_experience"),
    McpToolDefinition("omni_memory_search_experiences"),
    McpToolDefinition("omni_memory_write_skill", _skill_props(), ["name"]),
    McpToolDefinition("omni_memory_list_skills"),
    McpToolDefinition("omni_memory_get_skill"),
    McpToolDefinition("omni_memory_search_skills"),
    McpToolDefinition("omni_memory_write_failure_pattern"),
    McpToolDefinition("omni_memory_list_failure_patterns"),
    McpToolDefinition("omni_memory_get_failure_pattern"),
    McpToolDefinition("omni_memory_search_failure_patterns"),
    McpToolDefinition("omni_memory_consolidate_experiences"),
    McpToolDefinition("omni_memory_record_agent_cycle"),
    McpToolDefinition("omni_memory_draft_development_cycle", _development_cycle_props(), ["goal"]),
    McpToolDefinition("omni_memory_record_development_cycle", _development_cycle_props(), ["goal", "lesson"]),
    McpToolDefinition("omni_memory_finish_development_task", _finish_task_props(), ["goal", "lesson"]),
    McpToolDefinition("omni_memory_draft_ops_cycle", _ops_cycle_props(), ["goal", "service"]),
    McpToolDefinition("omni_memory_record_ops_cycle", _ops_cycle_props(), ["goal", "service", "lesson"]),
    McpToolDefinition("omni_memory_submit_review_item", _review_item_props(), ["kind", "title", "payload"]),
    McpToolDefinition("omni_memory_list_review_items", _review_list_props()),
    McpToolDefinition("omni_memory_get_review_item", {"item_id": {"type": "string"}}, ["item_id"]),
    McpToolDefinition("omni_memory_accept_review_item", _review_action_props(), ["item_id"]),
    McpToolDefinition("omni_memory_reject_review_item", _review_action_props(), ["item_id"]),
    McpToolDefinition(
        "omni_memory_supersede_review_item",
        {**_review_action_props(), "replacement": {"type": "object"}},
        ["item_id", "replacement"],
    ),
    McpToolDefinition("omni_memory_session_ingest_turn"),
    McpToolDefinition("omni_memory_session_commit", _session_commit_props()),
    McpToolDefinition("omni_memory_session_clear"),
    McpToolDefinition("omni_memory_clear"),
    McpToolDefinition("omni_memory_stats"),
]
