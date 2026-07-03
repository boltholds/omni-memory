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
        name="omni_memory_list_facts",
        description="List stored facts with optional filters.",
        structured_output=False,
    )
    def list_facts(
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> str:
        return call(
            "omni_memory_list_facts",
            subject=subject,
            predicate=predicate,
            object=object,
            status=status,
            limit=limit,
        )

    @server.tool(
        name="omni_memory_get_fact",
        description="Get a stored fact by id.",
        structured_output=False,
    )
    def get_fact(fact_id: str) -> str:
        return call("omni_memory_get_fact", fact_id=fact_id)

    @server.tool(
        name="omni_memory_patch_fact",
        description="Patch a stored fact in place through fact maintenance strategies.",
        structured_output=False,
    )
    def patch_fact(
        fact_id: str,
        patch: dict[str, Any],
        reason: str | None = None,
        dry_run: bool = False,
    ) -> str:
        return call(
            "omni_memory_patch_fact",
            fact_id=fact_id,
            patch=patch,
            reason=reason,
            dry_run=dry_run,
        )

    @server.tool(
        name="omni_memory_retract_fact",
        description="Soft-delete a fact by marking it retracted.",
        structured_output=False,
    )
    def retract_fact(
        fact_id: str,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> str:
        return call(
            "omni_memory_retract_fact",
            fact_id=fact_id,
            reason=reason,
            dry_run=dry_run,
        )

    @server.tool(
        name="omni_memory_supersede_fact",
        description="Create a new current fact and mark an old fact historical.",
        structured_output=False,
    )
    def supersede_fact(
        fact_id: str,
        new_fact: dict[str, Any],
        reason: str | None = None,
        source: str = "mcp",
        dry_run: bool = False,
    ) -> str:
        return call(
            "omni_memory_supersede_fact",
            fact_id=fact_id,
            new_fact=new_fact,
            reason=reason,
            source=source,
            dry_run=dry_run,
        )

    @server.tool(
        name="omni_memory_delete_fact",
        description="Delete a fact. Soft delete by default; hard=true removes storage record.",
        structured_output=False,
    )
    def delete_fact(
        fact_id: str,
        hard: bool = False,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> str:
        return call(
            "omni_memory_delete_fact",
            fact_id=fact_id,
            hard=hard,
            reason=reason,
            dry_run=dry_run,
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
        name="omni_memory_write_decision",
        description="Save a project decision/ADR record.",
        structured_output=False,
    )
    def write_decision(
        title: str,
        decision: str,
        context: str = "",
        consequences: list[str] | None = None,
        alternatives: list[str] | None = None,
        refs: dict[str, Any] | None = None,
        status: str = "accepted",
        source: str = "mcp",
        meta: dict[str, Any] | None = None,
    ) -> str:
        return call(
            "omni_memory_write_decision",
            title=title,
            decision=decision,
            context=context,
            consequences=consequences or [],
            alternatives=alternatives or [],
            refs=refs or {},
            status=status,
            source=source,
            meta=meta or {},
        )

    @server.tool(
        name="omni_memory_list_decisions",
        description="List project decision/ADR records.",
        structured_output=False,
    )
    def list_decisions(status: str | None = None, limit: int | None = None) -> str:
        return call("omni_memory_list_decisions", status=status, limit=limit)

    @server.tool(
        name="omni_memory_get_decision",
        description="Get a project decision/ADR record by id.",
        structured_output=False,
    )
    def get_decision(decision_id: str) -> str:
        return call("omni_memory_get_decision", decision_id=decision_id)

    @server.tool(
        name="omni_memory_write_experience",
        description="Save an agent experience record: goal, action, outcome, lesson and reuse conditions.",
        structured_output=False,
    )
    def write_experience(
        goal: str,
        lesson: str,
        context: str = "",
        decision: str = "",
        actions: list[str] | None = None,
        outcome: str = "",
        evaluation: dict[str, Any] | None = None,
        reuse_when: list[str] | None = None,
        avoid_when: list[str] | None = None,
        confidence: float = 0.5,
        refs: dict[str, Any] | None = None,
        source: str = "mcp",
        meta: dict[str, Any] | None = None,
    ) -> str:
        return call(
            "omni_memory_write_experience",
            goal=goal,
            lesson=lesson,
            context=context,
            decision=decision,
            actions=actions or [],
            outcome=outcome,
            evaluation=evaluation or {},
            reuse_when=reuse_when or [],
            avoid_when=avoid_when or [],
            confidence=confidence,
            refs=refs or {},
            source=source,
            meta=meta or {},
        )

    @server.tool(
        name="omni_memory_list_experiences",
        description="List agent experience records.",
        structured_output=False,
    )
    def list_experiences(limit: int | None = None) -> str:
        return call("omni_memory_list_experiences", limit=limit)

    @server.tool(
        name="omni_memory_get_experience",
        description="Get an agent experience record by id.",
        structured_output=False,
    )
    def get_experience(experience_id: str) -> str:
        return call("omni_memory_get_experience", experience_id=experience_id)

    @server.tool(
        name="omni_memory_search_experiences",
        description="Search agent experience records by intent, lesson or reuse condition.",
        structured_output=False,
    )
    def search_experiences(query: str, k: int = 5) -> str:
        return call("omni_memory_search_experiences", query=query, k=k)

    @server.tool(
        name="omni_memory_record_agent_cycle",
        description="Record a completed agent cycle as reusable experience.",
        structured_output=False,
    )
    def record_agent_cycle(
        goal: str,
        lesson: str,
        plan: list[str] | None = None,
        decisions: list[str] | None = None,
        actions: list[str] | None = None,
        outcome: str = "",
        tests: list[str] | None = None,
        files: list[str] | None = None,
        side_effects: list[str] | None = None,
        reuse_when: list[str] | None = None,
        avoid_when: list[str] | None = None,
        confidence: float = 0.8,
        source: str = "mcp-agent-cycle",
        meta: dict[str, Any] | None = None,
    ) -> str:
        return call(
            "omni_memory_record_agent_cycle",
            goal=goal,
            lesson=lesson,
            plan=plan or [],
            decisions=decisions or [],
            actions=actions or [],
            outcome=outcome,
            tests=tests or [],
            files=files or [],
            side_effects=side_effects or [],
            reuse_when=reuse_when or [],
            avoid_when=avoid_when or [],
            confidence=confidence,
            source=source,
            meta=meta or {},
        )

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
        name="omni_memory_clear",
        description="Clear durable OmniMemory stores and/or the in-process session buffer.",
        structured_output=False,
    )
    def clear(
        include_vectors: bool = True,
        include_facts: bool = True,
        include_episodes: bool = True,
        include_decisions: bool = True,
        include_experiences: bool = True,
        include_session: bool = True,
        dry_run: bool = False,
    ) -> str:
        return call(
            "omni_memory_clear",
            include_vectors=include_vectors,
            include_facts=include_facts,
            include_episodes=include_episodes,
            include_decisions=include_decisions,
            include_experiences=include_experiences,
            include_session=include_session,
            dry_run=dry_run,
        )

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
