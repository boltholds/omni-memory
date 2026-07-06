from __future__ import annotations

import inspect
import json

import pytest

from omni_memory import build_memory
from omni_memory.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from omni_memory.integrations.mcp_registry import (
    MCP_PROFILE_AGENT_CORE,
    MCP_PROFILE_MAINTENANCE,
    MCP_PROFILES,
    MCP_TOOL_REGISTRY,
    mcp_tool_names_for_profile,
    mcp_tool_profiles,
)
from omni_memory.mcp_server import _build_tool_function, build_mcp_app
from omni_memory.infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_mcp_tool_schemas_include_core_memory_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}

    assert "omni_memory_write_items" in names
    assert "omni_memory_retrieve" in names
    assert "omni_memory_context" in names
    assert "omni_memory_detect_conflicts" in names
    assert "omni_memory_session_commit" in names
    assert "omni_memory_clear" in names
    assert "omni_memory_list_facts" in names
    assert "omni_memory_get_fact" in names
    assert "omni_memory_retract_fact" in names
    assert "omni_memory_supersede_fact" in names
    assert "omni_memory_write_decision" in names
    assert "omni_memory_list_decisions" in names
    assert "omni_memory_get_decision" in names
    assert "omni_memory_write_experience" in names
    assert "omni_memory_search_experiences" in names
    assert "omni_memory_record_agent_cycle" in names
    assert "omni_memory_stats" in names


def test_mcp_write_tool_schemas_describe_required_inputs():
    by_name = {tool["name"]: tool for tool in MCP_TOOL_SCHEMAS}

    write_fact = by_name["omni_memory_write_fact"]["inputSchema"]
    assert write_fact["required"] == ["subject", "predicate", "object"]
    assert {"subject", "predicate", "object"} <= set(write_fact["properties"])

    write_skill = by_name["omni_memory_write_skill"]["inputSchema"]
    assert write_skill["required"] == ["name"]
    assert {"name", "procedure", "reuse_when"} <= set(write_skill["properties"])

    session_commit = by_name["omni_memory_session_commit"]["inputSchema"]
    assert {"dry_run", "min_confidence", "clear", "meta"} <= set(session_commit["properties"])


def test_mcp_registry_describes_runtime_handler_inputs():
    by_name = {tool["name"]: tool for tool in MCP_TOOL_SCHEMAS}

    write_items = by_name["omni_memory_write_items"]["inputSchema"]
    assert write_items["required"] == ["items"]
    assert {"items", "source", "dry_run"} <= set(write_items["properties"])

    get_fact = by_name["omni_memory_get_fact"]["inputSchema"]
    assert get_fact["required"] == ["fact_id"]

    finish_task = by_name["omni_memory_finish_development_task"]["inputSchema"]
    assert finish_task["required"] == ["goal", "lesson"]
    assert {
        "goal",
        "lesson",
        "changed_files",
        "commands_run",
        "tests",
        "run_distiller",
        "distill_dry_run",
        "clear_session",
    } <= set(finish_task["properties"])

    review_action = by_name["omni_memory_accept_review_item"]["inputSchema"]
    assert review_action["required"] == ["item_id"]
    assert {"reviewer", "note"} <= set(review_action["properties"])

    session_turn = by_name["omni_memory_session_ingest_turn"]["inputSchema"]
    assert session_turn["required"] == ["role", "content"]


def test_mcp_server_builds_tools_from_registry_contracts():
    handlers = build_mcp_handlers(_memory())
    registry_names = {definition.name for definition in MCP_TOOL_REGISTRY}

    assert registry_names == set(handlers)
    assert build_mcp_app(_memory()) is not None


def test_mcp_tool_function_signature_is_generated_from_registry():
    definitions = {definition.name: definition for definition in MCP_TOOL_REGISTRY}
    handler_calls = []

    def fake_handler(**kwargs):
        handler_calls.append(kwargs)
        return {"ok": True, "kwargs": kwargs}

    tool = _build_tool_function(definitions["omni_memory_finish_development_task"], fake_handler)
    signature = inspect.signature(tool)

    assert list(signature.parameters)[:2] == ["goal", "lesson"]
    assert signature.parameters["goal"].default is inspect.Parameter.empty
    assert signature.parameters["lesson"].default is inspect.Parameter.empty
    assert signature.parameters["changed_files"].default == []
    assert signature.parameters["run_distiller"].default is True

    payload = json.loads(tool(goal="Refactor MCP", lesson="Use registry-driven tools."))

    assert payload["ok"] is True
    assert handler_calls[0]["goal"] == "Refactor MCP"
    assert handler_calls[0]["lesson"] == "Use registry-driven tools."
    assert handler_calls[0]["changed_files"] == []
    assert handler_calls[0]["run_distiller"] is True


def test_mcp_advertised_tools_are_callable():
    advertised = {tool["name"] for tool in MCP_TOOL_SCHEMAS}
    handlers = build_mcp_handlers(_memory())

    assert advertised == set(handlers)
    assert all(callable(handler) for handler in handlers.values())


def test_mcp_tool_profiles_expose_small_agent_core_surface():
    profiles = mcp_tool_profiles()

    assert set(profiles) == set(MCP_PROFILES)
    assert profiles[MCP_PROFILE_AGENT_CORE] == [
        "omni_memory_retrieve",
        "omni_memory_context",
        "omni_memory_search_experiences",
        "omni_memory_search_failure_patterns",
        "omni_memory_finish_development_task",
    ]
    assert "omni_memory_clear" in profiles[MCP_PROFILE_MAINTENANCE]
    assert "omni_memory_accept_review_item" in profiles[MCP_PROFILE_MAINTENANCE]
    assert "omni_memory_list_facts" in profiles[MCP_PROFILE_MAINTENANCE]
    assert "omni_memory_session_commit" in profiles[MCP_PROFILE_MAINTENANCE]
    assert "omni_memory_consolidate_experiences" in profiles[MCP_PROFILE_MAINTENANCE]


def test_every_mcp_tool_has_a_supported_profile():
    names_by_profile = set()

    for definition in MCP_TOOL_REGISTRY:
        assert definition.profile in MCP_PROFILES
        names_by_profile.add(definition.name)

    assert names_by_profile == set(mcp_tool_names_for_profile(MCP_PROFILE_AGENT_CORE)) | set(
        mcp_tool_names_for_profile(MCP_PROFILE_MAINTENANCE)
    )


def test_mcp_handlers_write_retrieve_context_and_conflicts():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    assert write["saved"] == 1

    retrieve = handlers["omni_memory_retrieve"](query="Where is Alice?")
    assert any(fact["object"] == "lighthouse" for fact in retrieve["facts"])

    context = handlers["omni_memory_context"](query="Where is Alice?")
    assert any(section["title"] in {"Facts", "Current Beliefs"} for section in context["sections"])

    conflicts = handlers["omni_memory_detect_conflicts"](
        facts=[
            {"id": "f1", "subject": "alice", "predicate": "at", "object": "lighthouse"},
            {"id": "f2", "subject": "alice", "predicate": "at", "object": "bridge"},
        ]
    )
    assert conflicts["conflicts"][0]["key"] == "alice::at"


def test_mcp_clear_removes_selected_memory_stores():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    handlers["omni_memory_write_note"](text="Project uses MCP tools.", source="test")
    handlers["omni_memory_session_ingest_turn"](role="user", content="remember this transiently")

    dry_run = handlers["omni_memory_clear"](dry_run=True)
    assert dry_run == {
        "vector_objects": 1,
        "facts": 1,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "skills": 0,
        "failure_patterns": 0,
        "review_items": 0,
        "session_turns": 1,
        "dry_run": True,
    }
    assert handlers["omni_memory_stats"]()["facts"] == 1

    cleared = handlers["omni_memory_clear"]()
    assert cleared == {
        "vector_objects": 1,
        "facts": 1,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "skills": 0,
        "failure_patterns": 0,
        "review_items": 0,
        "session_turns": 1,
        "dry_run": False,
    }

    stats = handlers["omni_memory_stats"]()
    assert stats["vector_objects"] == 0
    assert stats["facts"] == 0
    assert stats["decisions"] == 0
    assert stats["experiences"] == 0
    assert stats["skills"] == 0
    assert stats["failure_patterns"] == 0
    assert stats["session_turns"] == 0
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["facts"] == []
    assert handlers["omni_memory_retrieve"](query="alice lighthouse project")["semantic_chunks"] == []


def test_mcp_clear_preserves_excluded_memory_stores():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_fact"](
        subject="alice",
        predicate="at",
        object="lighthouse",
        source="test",
    )
    handlers["omni_memory_write_note"](text="Project uses MCP tools.", source="test")
    handlers["omni_memory_session_ingest_turn"](role="user", content="transient turn")

    cleared = handlers["omni_memory_clear"](
        include_facts=False,
        include_episodes=False,
        include_decisions=False,
        include_experiences=False,
    )
    assert cleared == {
        "vector_objects": 1,
        "facts": 0,
        "episodes": 0,
        "decisions": 0,
        "experiences": 0,
        "skills": 0,
        "failure_patterns": 0,
        "review_items": 0,
        "session_turns": 1,
        "dry_run": False,
    }

    stats = handlers["omni_memory_stats"]()
    assert stats["vector_objects"] == 0
    assert stats["facts"] == 1
    assert stats["skills"] == 0
    assert stats["failure_patterns"] == 0
    assert stats["session_turns"] == 0

    retrieved = handlers["omni_memory_retrieve"](query="Where is Alice?")
    assert any(fact["object"] == "lighthouse" for fact in retrieved["facts"])
    assert handlers["omni_memory_retrieve"](query="Project uses MCP tools")["semantic_chunks"] == []


def test_mcp_fact_maintenance_supersedes_without_creating_current_conflict():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_items"](
        items=[
            {
                "id": "fact-mcp-old",
                "type": "fact",
                "subject": "mcp_server",
                "predicate": "implemented_with",
                "object": "minimal json-rpc",
                "meta": {"confidence": 0.7},
            }
        ],
        source="test",
    )
    assert write["saved"] == 1

    supersede = handlers["omni_memory_supersede_fact"](
        fact_id="fact-mcp-old",
        new_fact={
            "id": "fact-mcp-new",
            "object": "official MCP SDK FastMCP",
            "meta": {"confidence": 0.99},
        },
        reason="Migrated to official MCP SDK",
        source="test",
    )
    assert supersede["applied"] is True
    assert supersede["fact"]["id"] == "fact-mcp-new"

    old = handlers["omni_memory_get_fact"](fact_id="fact-mcp-old")["fact"]
    assert old["meta"]["status"] == "historical"
    assert old["meta"]["superseded_by"] == "fact-mcp-new"

    current = handlers["omni_memory_list_facts"](status="current")["facts"]
    historical = handlers["omni_memory_list_facts"](status="historical")["facts"]
    assert [fact["id"] for fact in current] == ["fact-mcp-new"]
    assert [fact["id"] for fact in historical] == ["fact-mcp-old"]

    all_facts = handlers["omni_memory_list_facts"]()["facts"]
    conflicts = handlers["omni_memory_detect_conflicts"](facts=all_facts)
    assert conflicts["conflicts"] == []

    retrieved = handlers["omni_memory_retrieve"](query="mcp_server implementation", k_sem=1)
    belief = retrieved["beliefs"][0]
    assert belief["current"]["id"] == "fact-mcp-new"
    assert [fact["id"] for fact in belief["historical"]] == ["fact-mcp-old"]


def test_mcp_fact_maintenance_retracts_fact_from_current_beliefs():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_items"](
        items=[
            {
                "id": "fact-temp",
                "type": "fact",
                "subject": "temporary_fact",
                "predicate": "status",
                "object": "active",
            }
        ],
        source="test",
    )

    retracted = handlers["omni_memory_retract_fact"](
        fact_id="fact-temp",
        reason="No longer true",
    )
    assert retracted["applied"] is True
    assert retracted["fact"]["meta"]["status"] == "retracted"

    retrieved = handlers["omni_memory_retrieve"](query="temporary_fact status", k_sem=1)
    assert retrieved["beliefs"] == []


def test_mcp_delete_fact_hard_removes_fact_record():
    handlers = build_mcp_handlers(_memory())
    handlers["omni_memory_write_items"](
        items=[
            {
                "id": "fact-hard-delete",
                "type": "fact",
                "subject": "hard_delete_fact",
                "predicate": "status",
                "object": "temporary",
                "meta": {"confidence": 1.0},
            }
        ],
        source="test",
    )

    deleted = handlers["omni_memory_delete_fact"](fact_id="fact-hard-delete", hard=True)

    assert deleted["operation"] == "hard_delete"
    assert deleted["removed"] == 1
    assert handlers["omni_memory_get_fact"](fact_id="fact-hard-delete")["fact"] is None


def test_mcp_decision_records_are_written_listed_and_retrieved_in_context():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_decision"](
        title="Use FastMCP for MCP server",
        decision="Use official MCP SDK FastMCP instead of a handwritten JSON-RPC loop.",
        context="MCP clients expect protocol-compatible stdio behavior.",
        consequences=["Better compatibility with Codex and other MCP clients."],
        alternatives=["Keep minimal newline-delimited JSON-RPC server."],
        refs={"files": ["app/mcp_server.py"]},
        source="test",
    )
    assert write["saved"] == 1

    listed = handlers["omni_memory_list_decisions"](limit=5)["decisions"]
    assert listed[0]["title"] == "Use FastMCP for MCP server"

    decision_id = listed[0]["id"]
    got = handlers["omni_memory_get_decision"](decision_id=decision_id)["decision"]
    assert got["decision"].startswith("Use official MCP SDK")

    context = handlers["omni_memory_context"](
        query="Why did we choose FastMCP?",
        intent="make_decision",
    )
    sections = {section["title"]: section["body"] for section in context["sections"]}
    assert "Decision Records" in sections
    assert "Use FastMCP for MCP server" in sections["Decision Records"]


def test_mcp_experience_records_are_searchable_and_retrieved_in_context():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_write_experience"](
        goal="Improve MCP compatibility",
        decision="Use official protocol SDK",
        actions=["Replaced handwritten JSON-RPC loop with FastMCP"],
        outcome="Codex could discover MCP tools after restart.",
        evaluation={"tests": "passed", "success": True},
        lesson="Prefer the official protocol SDK when MCP client compatibility matters.",
        reuse_when=["building MCP integrations", "protocol compatibility matters"],
        avoid_when=["dependency footprint must stay minimal"],
        confidence=0.95,
        refs={"files": ["app/mcp_server.py"]},
        source="test",
    )
    assert write["saved"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="MCP compatibility protocol SDK",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Improve MCP compatibility"
    assert "protocol SDK" in found[0]["lesson"]

    context = handlers["omni_memory_context"](
        query="How should we improve MCP compatibility?",
        intent="debug_failure",
    )
    sections = {section["title"]: section["body"] for section in context["sections"]}
    assert "Relevant Experience" in sections
    assert "Prefer the official protocol SDK" in sections["Relevant Experience"]
    assert "building MCP integrations" in sections["Relevant Experience"]


def test_mcp_context_uses_memory_intent_to_select_relevant_sections():
    handlers = build_mcp_handlers(_memory())

    handlers["omni_memory_write_fact"](
        subject="omnimemory",
        predicate="framework",
        object="fastapi",
        source="test",
    )
    handlers["omni_memory_write_note"](
        text="OmniMemory has a public FastAPI context endpoint.",
        source="test",
    )
    handlers["omni_memory_write_decision"](
        title="Use command interpreter for writes",
        decision="Keep OmniMemory as facade and execute write intents through command objects.",
        source="test",
    )
    handlers["omni_memory_write_experience"](
        goal="Refactor OmniMemory facade",
        lesson="When facade methods build raw payloads, move intent translation into commands.",
        reuse_when=["facade accumulates write operations"],
        source="test",
    )

    answer_context = handlers["omni_memory_context"](
        query="How does OmniMemory write memory?",
        intent="answer_question",
    )
    answer_titles = {section["title"] for section in answer_context["sections"]}
    assert "Facts" in answer_titles or "Current Beliefs" in answer_titles
    assert "Semantic Notes" in answer_titles
    assert "Decision Records" not in answer_titles
    assert "Relevant Experience" not in answer_titles

    code_context = handlers["omni_memory_context"](
        query="How should we refactor OmniMemory write operations?",
        intent="write_code",
    )
    code_sections = {section["title"]: section["body"] for section in code_context["sections"]}
    assert set(code_sections) == {"Decision Records", "Relevant Experience", "Semantic Notes"}
    assert "command objects" in code_sections["Decision Records"]
    assert "move intent translation into commands" in code_sections["Relevant Experience"]

    retrieved = handlers["omni_memory_retrieve"](
        query="How should we refactor OmniMemory write operations?",
        intent="make_decision",
    )
    assert retrieved["facts"] == []
    assert retrieved["semantic_chunks"] == []
    assert retrieved["decisions"]
    assert retrieved["experiences"]


def test_mcp_agent_cycle_records_reusable_experience():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_record_agent_cycle"](
        goal="Make fact maintenance manageable",
        plan=["Keep repositories as storage primitives", "Move semantic operations into strategies"],
        decisions=["Use FactMaintenanceService with strategies"],
        actions=["Added list/get/retract/supersede strategies", "Exposed MCP maintenance tools"],
        outcome="Fact maintenance works without making GraphRepo responsible for business semantics.",
        tests=["71 passed"],
        files=["app/fact_maintenance.py", "infra/repo/graph_repo.py"],
        side_effects=["MCP server must be restarted for new tools"],
        lesson="Maintenance behavior belongs in strategies, not storage repositories.",
        reuse_when=["adding semantic operations over persisted memory"],
        avoid_when=["pure storage serialization changes"],
        confidence=0.91,
        source="test-agent-cycle",
        meta={"task_id": "mcp-maintenance"},
    )

    assert write["saved"] == 1
    exp = write["experience"]
    assert exp["goal"] == "Make fact maintenance manageable"
    assert exp["decision"] == "Use FactMaintenanceService with strategies"
    assert "Exposed MCP maintenance tools" in exp["actions"]
    assert exp["evaluation"]["tests"] == ["71 passed"]
    assert exp["refs"]["files"] == ["app/fact_maintenance.py", "infra/repo/graph_repo.py"]
    assert exp["meta"]["domain"] == "development"
    assert exp["meta"]["task_id"] == "mcp-maintenance"

    found = handlers["omni_memory_search_experiences"](
        query="where should fact maintenance behavior live",
        k=3,
    )["experiences"]
    assert found[0]["id"] == exp["id"]


@pytest.mark.parametrize(
    "tool_name",
    [
        "omni_memory_record_development_cycle",
        "omni_memory_finish_development_task",
    ],
)
def test_mcp_development_workflow_records_reusable_experience(tool_name: str):
    handlers = build_mcp_handlers(_memory())

    write = handlers[tool_name](
        goal="Refactor MCP server",
        summary="Moved MCP tools behind workflow helpers.",
        changed_files=["app/mcp_server.py", "app/integrations/mcp.py"],
        commands_run=["pytest tests/test_mcp_server.py -q"],
        tests=["tests/test_mcp_server.py passed"],
        decisions=["Keep MCP API stable"],
        outcome="MCP workflow tool records development experience.",
        lesson="Record completed development work through the workflow helper instead of raw notes.",
        reuse_when=["finishing coding-agent tasks"],
        avoid_when=["temporary scratchpad notes"],
        confidence=0.92,
        source="test-development-workflow",
        meta={"phase": "mcp"},
        run_distiller=False,
    )

    assert write["experience"]["saved"] == 1 if tool_name == "omni_memory_finish_development_task" else write["saved"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="workflow helper raw notes coding-agent tasks",
        k=3,
    )["experiences"]
    assert found
    assert found[0]["goal"] == "Refactor MCP server"


def test_mcp_ops_cycle_records_searchable_experience():
    handlers = build_mcp_handlers(_memory())

    write = handlers["omni_memory_record_ops_cycle"](
        goal="Reduce API latency",
        service="memory-api",
        alert_id="latency-123",
        symptoms=["p95 latency above threshold"],
        actions=["reduced retrieval fanout", "added cache"],
        outcome="p95 latency recovered",
        metrics_before={"p95_ms": 1200.0},
        metrics_after={"p95_ms": 300.0},
        lesson="When retrieval fanout spikes latency, reduce fanout before adding new infrastructure.",
        reuse_when=["p95 latency regression in memory-api"],
        avoid_when=["database outage"],
        affected_resources=["retriever", "cache"],
        confidence=0.9,
        source="test-ops-cycle",
    )

    assert write["saved"] == 1
    exp = write["experience"]
    assert exp["goal"] == "Reduce API latency"
    assert exp["meta"]["domain"] == "ops"
    assert exp["refs"]["affected_resources"] == ["retriever", "cache"]

    found = handlers["omni_memory_search_experiences"](
        query="p95 latency regression fanout memory-api",
        k=3,
    )["experiences"]
    assert found[0]["id"] == exp["id"]
