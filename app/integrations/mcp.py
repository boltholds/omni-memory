from __future__ import annotations

from typing import Any, Callable

from app.integrations.fact_mining_mcp import FACT_MINING_TOOL_SCHEMA, build_fact_mining_handler
from app.memory import OmniMemory
from domain.models import Fact, FailurePatternRecord, SkillRecord


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "object", "properties": props or {}}
    if required:
        out["required"] = required
    return out


def _scope_schema() -> dict[str, Any]:
    return _schema(
        {
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
    )


def _tool(name: str, props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "description": name.replace("_", " "), "inputSchema": _schema(props, required)}


def _fact_props() -> dict[str, Any]:
    return {"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}, "source": {"type": "string", "default": "mcp"}, "confidence": {"type": "number", "default": 1.0}}


def _skill_props() -> dict[str, Any]:
    return {"name": {"type": "string"}, "problem": {"type": "string", "default": ""}, "procedure": {"type": "array", "items": {"type": "string"}, "default": []}, "reuse_when": {"type": "array", "items": {"type": "string"}, "default": []}, "avoid_when": {"type": "array", "items": {"type": "string"}, "default": []}, "evidence_ids": {"type": "array", "items": {"type": "string"}, "default": []}, "confidence": {"type": "number", "default": 0.5}, "source": {"type": "string", "default": "mcp"}, "meta": {"type": "object", "default": {}}}


def _session_commit_props() -> dict[str, Any]:
    return {"source": {"type": "string", "default": "mcp-session"}, "dry_run": {"type": "boolean", "default": False}, "min_confidence": {"type": "number", "default": 0.75}, "clear": {"type": "boolean", "default": True}, "meta": {"type": "object", "default": {}}}


def _finish_task_props() -> dict[str, Any]:
    props = _development_cycle_schema_props()
    props.update({"source": {"type": "string", "default": "mcp-development-workflow"}, "session_turns": {"type": "array", "items": {"type": "object"}, "default": []}, "run_distiller": {"type": "boolean", "default": True}, "distill_dry_run": {"type": "boolean", "default": True}, "min_confidence": {"type": "number", "default": 0.75}, "clear_session": {"type": "boolean", "default": False}})
    return props


def _development_cycle_schema_props() -> dict[str, Any]:
    return {"goal": {"type": "string"}, "summary": {"type": "string", "default": ""}, "changed_files": {"type": "array", "items": {"type": "string"}, "default": []}, "commands_run": {"type": "array", "items": {"type": "string"}, "default": []}, "tests": {"type": "array", "items": {"type": "string"}, "default": []}, "decisions": {"type": "array", "items": {"type": "string"}, "default": []}, "outcome": {"type": "string", "default": ""}, "lesson": {"type": "string", "default": ""}, "reuse_when": {"type": "array", "items": {"type": "string"}, "default": []}, "avoid_when": {"type": "array", "items": {"type": "string"}, "default": []}, "side_effects": {"type": "array", "items": {"type": "string"}, "default": []}, "confidence": {"type": "number", "default": 0.8}, "meta": {"type": "object", "default": {}}}


_NAMES = [
    "omni_memory_write_items", "omni_memory_retrieve", "omni_memory_ask", "omni_memory_context", "omni_memory_detect_conflicts", "omni_memory_mine_facts",
    "omni_memory_write_fact", "omni_memory_list_facts", "omni_memory_get_fact", "omni_memory_patch_fact", "omni_memory_retract_fact", "omni_memory_supersede_fact", "omni_memory_delete_fact",
    "omni_memory_write_note", "omni_memory_write_decision", "omni_memory_list_decisions", "omni_memory_get_decision",
    "omni_memory_write_experience", "omni_memory_list_experiences", "omni_memory_get_experience", "omni_memory_search_experiences",
    "omni_memory_write_skill", "omni_memory_list_skills", "omni_memory_get_skill", "omni_memory_search_skills",
    "omni_memory_write_failure_pattern", "omni_memory_list_failure_patterns", "omni_memory_get_failure_pattern", "omni_memory_search_failure_patterns",
    "omni_memory_consolidate_experiences", "omni_memory_record_agent_cycle", "omni_memory_draft_development_cycle", "omni_memory_record_development_cycle", "omni_memory_finish_development_task",
    "omni_memory_session_ingest_turn", "omni_memory_session_commit", "omni_memory_session_clear", "omni_memory_clear", "omni_memory_stats",
]

MCP_TOOL_SCHEMAS = [_tool(name) for name in _NAMES]
_BY_NAME = {item["name"]: item for item in MCP_TOOL_SCHEMAS}
_BY_NAME["omni_memory_retrieve"] = _tool("omni_memory_retrieve", {"query": {"type": "string"}, "k_sem": {"type": "integer", "default": 5}, "k_eps": {"type": "integer", "default": 3}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()}, ["query"])
_BY_NAME["omni_memory_ask"] = _tool("omni_memory_ask", {"question": {"type": "string"}, "lang": {"type": "string", "default": "en"}, "style": {"type": "string", "default": "concise"}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()}, ["question"])
_BY_NAME["omni_memory_context"] = _tool("omni_memory_context", {"query": {"type": "string", "default": ""}, "intent": {"type": "string"}, "mode": {"type": "string"}, "scope": _scope_schema()})
_BY_NAME["omni_memory_detect_conflicts"] = _tool("omni_memory_detect_conflicts", {"query": {"type": "string"}, "facts": {"type": "array", "items": {"type": "object"}}, "scope": _scope_schema()})
_BY_NAME["omni_memory_mine_facts"] = FACT_MINING_TOOL_SCHEMA
_BY_NAME["omni_memory_write_fact"] = _tool("omni_memory_write_fact", _fact_props(), ["subject", "predicate", "object"])
_BY_NAME["omni_memory_write_skill"] = _tool("omni_memory_write_skill", _skill_props(), ["name"])
_BY_NAME["omni_memory_session_commit"] = _tool("omni_memory_session_commit", _session_commit_props())
_BY_NAME["omni_memory_draft_development_cycle"] = _tool("omni_memory_draft_development_cycle", _development_cycle_schema_props(), ["goal"])
_BY_NAME["omni_memory_record_development_cycle"] = _tool("omni_memory_record_development_cycle", _development_cycle_schema_props(), ["goal", "lesson"])
_BY_NAME["omni_memory_finish_development_task"] = _tool("omni_memory_finish_development_task", _finish_task_props(), ["goal", "lesson"])
MCP_TOOL_SCHEMAS = [_BY_NAME[name] for name in _NAMES]


def build_mcp_handlers(memory: OmniMemory) -> dict[str, Callable[..., Any]]:
    def detect_conflicts(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("facts") is not None:
            return memory.consistency.detect_conflicts([Fact.model_validate(item) for item in kwargs["facts"]]).model_dump()
        return memory.detect_conflicts(kwargs.get("query"), scope=kwargs.get("scope") or {}).model_dump()

    return {
        "omni_memory_write_items": lambda **kw: memory.write_items(kw["items"], source=kw.get("source", "mcp"), dry_run=kw.get("dry_run", False)).model_dump(),
        "omni_memory_retrieve": lambda **kw: memory.retrieve(kw["query"], k_sem=kw.get("k_sem", 5), k_eps=kw.get("k_eps", 3), intent=kw.get("intent"), mode=kw.get("mode"), scope=kw.get("scope") or {}).model_dump(),
        "omni_memory_ask": lambda **kw: memory.ask(kw["question"], lang=kw.get("lang", "en"), style=kw.get("style", "concise"), intent=kw.get("intent"), mode=kw.get("mode"), scope=kw.get("scope") or {}).__dict__,
        "omni_memory_context": lambda **kw: memory.build_context(kw.get("query", ""), intent=kw.get("intent"), mode=kw.get("mode"), scope=kw.get("scope") or {}).model_dump(),
        "omni_memory_detect_conflicts": detect_conflicts,
        "omni_memory_mine_facts": build_fact_mining_handler(memory),
        "omni_memory_write_fact": lambda **kw: memory.write_fact(kw["subject"], kw["predicate"], kw["object"], source=kw.get("source", "mcp"), confidence=kw.get("confidence", 1.0)).model_dump(),
        "omni_memory_list_facts": lambda **kw: memory.maintain_facts({"operation": "list", "subject": kw.get("subject"), "predicate": kw.get("predicate"), "object": kw.get("object"), "status": kw.get("status"), "limit": kw.get("limit")}).model_dump(mode="json"),
        "omni_memory_get_fact": lambda **kw: memory.maintain_facts({"operation": "get", "fact_id": kw["fact_id"]}).model_dump(mode="json"),
        "omni_memory_patch_fact": lambda **kw: memory.maintain_facts({"operation": "patch", "fact_id": kw["fact_id"], "patch": kw.get("patch") or {}, "reason": kw.get("reason"), "dry_run": kw.get("dry_run", False)}).model_dump(mode="json"),
        "omni_memory_retract_fact": lambda **kw: memory.maintain_facts({"operation": "retract", "fact_id": kw["fact_id"], "reason": kw.get("reason"), "dry_run": kw.get("dry_run", False)}).model_dump(mode="json"),
        "omni_memory_supersede_fact": lambda **kw: memory.maintain_facts({"operation": "supersede", "fact_id": kw["fact_id"], "new_fact": kw.get("new_fact") or {}, "reason": kw.get("reason"), "source": kw.get("source", "mcp"), "dry_run": kw.get("dry_run", False)}).model_dump(mode="json"),
        "omni_memory_delete_fact": lambda **kw: memory.maintain_facts({"operation": "hard_delete" if kw.get("hard", False) else "retract", "fact_id": kw["fact_id"], "reason": kw.get("reason"), "dry_run": kw.get("dry_run", False)}).model_dump(mode="json"),
        "omni_memory_write_note": lambda **kw: memory.write_note(kw["text"], source=kw.get("source", "mcp"), meta=kw.get("meta") or {}).model_dump(),
        "omni_memory_write_decision": lambda **kw: memory.write_decision(title=kw["title"], decision=kw["decision"], context=kw.get("context", ""), consequences=kw.get("consequences") or [], alternatives=kw.get("alternatives") or [], refs=kw.get("refs") or {}, status=kw.get("status", "accepted"), source=kw.get("source", "mcp"), meta=kw.get("meta") or {}).model_dump(),
        "omni_memory_list_decisions": lambda **kw: {"decisions": [item.model_dump(mode="json") for item in memory.list_decisions(status=kw.get("status"), limit=kw.get("limit"))]},
        "omni_memory_get_decision": lambda **kw: {"decision": (item.model_dump(mode="json") if (item := memory.get_decision(kw["decision_id"])) is not None else None)},
        "omni_memory_write_experience": lambda **kw: memory.record_experience(goal=kw["goal"], lesson=kw["lesson"], context=kw.get("context", ""), decision=kw.get("decision", ""), actions=kw.get("actions") or [], outcome=kw.get("outcome", ""), evaluation=kw.get("evaluation") or {}, reuse_when=kw.get("reuse_when") or [], avoid_when=kw.get("avoid_when") or [], confidence=kw.get("confidence", 0.5), refs=kw.get("refs") or {}, source=kw.get("source", "mcp"), meta=kw.get("meta") or {}).model_dump(),
        "omni_memory_list_experiences": lambda **kw: {"experiences": [item.model_dump(mode="json") for item in memory.list_experiences(limit=kw.get("limit"))]},
        "omni_memory_get_experience": lambda **kw: {"experience": (item.model_dump(mode="json") if (item := memory.get_experience(kw["experience_id"])) is not None else None)},
        "omni_memory_search_experiences": lambda **kw: {"experiences": [item.model_dump(mode="json") for item in memory.search_experiences(kw["query"], k=kw.get("k", 5))]},
        "omni_memory_write_skill": lambda **kw: _write_skill(memory, **kw),
        "omni_memory_list_skills": lambda **kw: {"skills": [item.model_dump(mode="json") for item in memory.repositories.skill.list_skills(limit=kw.get("limit"))]},
        "omni_memory_get_skill": lambda **kw: {"skill": (item.model_dump(mode="json") if (item := memory.repositories.skill.get_skill(kw["skill_id"])) is not None else None)},
        "omni_memory_search_skills": lambda **kw: {"skills": [item.model_dump(mode="json") for item in memory.repositories.skill.search(kw["query"], k=kw.get("k", 5))]},
        "omni_memory_write_failure_pattern": lambda **kw: _write_failure_pattern(memory, **kw),
        "omni_memory_list_failure_patterns": lambda **kw: {"failure_patterns": [item.model_dump(mode="json") for item in memory.repositories.failure_pattern.list_failure_patterns(limit=kw.get("limit"))]},
        "omni_memory_get_failure_pattern": lambda **kw: {"failure_pattern": (item.model_dump(mode="json") if (item := memory.repositories.failure_pattern.get_failure_pattern(kw["pattern_id"])) is not None else None)},
        "omni_memory_search_failure_patterns": lambda **kw: {"failure_patterns": [item.model_dump(mode="json") for item in memory.repositories.failure_pattern.search(kw["query"], k=kw.get("k", 5))]},
        "omni_memory_consolidate_experiences": lambda **kw: memory.consolidate_experiences(dry_run=kw.get("dry_run", True), min_confidence=kw.get("min_confidence", 0.85)).model_dump(mode="json"),
        "omni_memory_record_agent_cycle": lambda **kw: memory.record_agent_cycle({"goal": kw["goal"], "plan": kw.get("plan") or [], "decisions": kw.get("decisions") or [], "actions": kw.get("actions") or [], "outcome": kw.get("outcome", ""), "tests": kw.get("tests") or [], "files": kw.get("files") or [], "side_effects": kw.get("side_effects") or [], "lesson": kw["lesson"], "reuse_when": kw.get("reuse_when") or [], "avoid_when": kw.get("avoid_when") or [], "confidence": kw.get("confidence", 0.8), "meta": kw.get("meta") or {}}, source=kw.get("source", "mcp-agent-cycle")).model_dump(),
        "omni_memory_draft_development_cycle": lambda **kw: memory.draft_development_cycle(_development_cycle_payload(kw)).model_dump(mode="json"),
        "omni_memory_record_development_cycle": lambda **kw: memory.record_development_cycle(_development_cycle_payload(kw), source=kw.get("source", "mcp-development-cycle")).model_dump(),
        "omni_memory_finish_development_task": lambda **kw: memory.development_memory_workflow.finish_task(_finish_development_task_payload(kw)).model_dump(mode="json"),
        "omni_memory_session_ingest_turn": lambda **kw: _session_ingest_turn(memory, role=kw["role"], content=kw["content"]),
        "omni_memory_session_commit": lambda **kw: memory.commit_session(source=kw.get("source", "mcp-session"), dry_run=kw.get("dry_run", False), meta=kw.get("meta") or {}, min_confidence=kw.get("min_confidence", 0.75), clear=kw.get("clear", True)).model_dump(),
        "omni_memory_session_clear": lambda **kw: _session_clear(memory),
        "omni_memory_clear": lambda **kw: memory.clear(include_vectors=kw.get("include_vectors", True), include_facts=kw.get("include_facts", True), include_episodes=kw.get("include_episodes", True), include_decisions=kw.get("include_decisions", True), include_experiences=kw.get("include_experiences", True), include_skills=kw.get("include_skills", True), include_failure_patterns=kw.get("include_failure_patterns", True), include_session=kw.get("include_session", True), dry_run=kw.get("dry_run", False)).__dict__,
        "omni_memory_stats": lambda **kw: _stats(memory),
    }


def _write_skill(memory: OmniMemory, **kw: Any) -> dict[str, Any]:
    result = memory.write_skill_raw(name=kw["name"], problem=kw.get("problem", ""), procedure=kw.get("procedure") or [], reuse_when=kw.get("reuse_when") or [], avoid_when=kw.get("avoid_when") or [], evidence_ids=kw.get("evidence_ids") or [], confidence=kw.get("confidence", 0.5), refs=kw.get("refs") or {}, source=kw.get("source", "mcp"), meta=kw.get("meta") or {})
    item = next((saved for saved in result.saved if isinstance(saved, SkillRecord)), None)
    return {"saved": result.saved_count, "rejected": result.rejected_count + result.error_count, "reasons": result.reasons, "skill": item.model_dump(mode="json") if item else None}


def _write_failure_pattern(memory: OmniMemory, **kw: Any) -> dict[str, Any]:
    result = memory.write_failure_pattern_raw(symptom=kw["symptom"], root_cause=kw.get("root_cause", ""), fix=kw.get("fix", ""), detection=kw.get("detection", ""), evidence_ids=kw.get("evidence_ids") or [], confidence=kw.get("confidence", 0.5), refs=kw.get("refs") or {}, source=kw.get("source", "mcp"), meta=kw.get("meta") or {})
    item = next((saved for saved in result.saved if isinstance(saved, FailurePatternRecord)), None)
    return {"saved": result.saved_count, "rejected": result.rejected_count + result.error_count, "reasons": result.reasons, "failure_pattern": item.model_dump(mode="json") if item else None}


def _development_cycle_payload(kw: dict[str, Any]) -> dict[str, Any]:
    return {"goal": kw["goal"], "summary": kw.get("summary", ""), "changed_files": kw.get("changed_files") or [], "commands_run": kw.get("commands_run") or [], "tests": kw.get("tests") or [], "decisions": kw.get("decisions") or [], "outcome": kw.get("outcome", ""), "lesson": kw.get("lesson", ""), "reuse_when": kw.get("reuse_when") or [], "avoid_when": kw.get("avoid_when") or [], "side_effects": kw.get("side_effects") or [], "confidence": kw.get("confidence", 0.8), "meta": kw.get("meta") or {}}


def _finish_development_task_payload(kw: dict[str, Any]) -> dict[str, Any]:
    payload = _development_cycle_payload(kw)
    payload.update({"source": kw.get("source", "mcp-development-workflow"), "session_turns": kw.get("session_turns") or [], "run_distiller": kw.get("run_distiller", True), "distill_dry_run": kw.get("distill_dry_run", True), "min_confidence": kw.get("min_confidence", 0.75), "clear_session": kw.get("clear_session", False)})
    return payload


def _session_ingest_turn(memory: OmniMemory, *, role: str, content: str) -> dict[str, Any]:
    memory.ingest_turn(role, content)
    return {"ok": True, "session_turns": len(memory._session_turns)}


def _session_clear(memory: OmniMemory) -> dict[str, Any]:
    memory.clear_session()
    return {"ok": True, "session_turns": 0}


def _stats(memory: OmniMemory) -> dict[str, Any]:
    return {**memory.repository_stats(), "session_turns": len(memory._session_turns), "llm_configured": memory.llm is not None}
