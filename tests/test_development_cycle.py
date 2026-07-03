from __future__ import annotations

import json

import pytest

from app.builder import build_memory
from app.development_cycle import DevelopmentCycleDraft
from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from app.mcp_server import build_mcp_app
from domain.distiller import MemoryCandidate, SessionDistillationResult, SessionTurn
from infra.embeddings.factory import HashEmbedder


class FakeWorkflowDistiller:
    def distill_session(self, turns: list[SessionTurn]) -> SessionDistillationResult:
        return SessionDistillationResult(
            candidates=[
                MemoryCandidate(
                    kind="fact",
                    should_write=True,
                    confidence=0.91,
                    reason="The implementation detail is explicit.",
                    evidence_quote="OmniMemory exposes a finish development task MCP tool.",
                    temporal_scope="current",
                    payload={
                        "subject": "OmniMemory",
                        "predicate": "exposes",
                        "object": "finish development task MCP tool",
                    },
                )
            ]
        )


def _memory(*, distiller=None):
    return build_memory(use_llm=False, embedder=HashEmbedder(), distiller=distiller)


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_development_cycle_draft_adapts_to_domain_neutral_agent_cycle():
    cycle = DevelopmentCycleDraft(
        goal="Generalize agent cycle",
        summary="Moved dev fields into domain adapter.",
        changed_files=["app/agent_cycle.py", "app/development_cycle.py"],
        commands_run=["poetry run pytest tests/test_development_cycle.py -q"],
        tests=["development cycle tests passed"],
        lesson="Development-specific cycle data should be adapted into a generic agent cycle.",
    ).to_agent_cycle()

    assert cycle.domain == "development"
    assert cycle.affected_resources == ["app/agent_cycle.py", "app/development_cycle.py"]
    assert cycle.validation["tests"] == ["development cycle tests passed"]
    assert cycle.validation["commands_run"] == ["poetry run pytest tests/test_development_cycle.py -q"]
    assert cycle.refs["files"] == ["app/agent_cycle.py", "app/development_cycle.py"]
    assert cycle.files == ["app/agent_cycle.py", "app/development_cycle.py"]
    assert cycle.meta["domain"] == "development"


def test_mcp_development_cycle_draft_does_not_write_memory():
    handlers = build_mcp_handlers(_memory())

    draft = handlers["omni_memory_draft_development_cycle"](
        goal="Wire development cycle recorder",
        summary="Added draft and record tools.",
        changed_files=["app/development_cycle.py", "app/integrations/mcp.py"],
        commands_run=["poetry run pytest -q tests/test_development_cycle.py"],
        tests=["development cycle tests passed"],
        decisions=["Draft first, record only when lesson is explicit"],
        reuse_when=["capturing completed development work"],
    )

    assert draft["goal"] == "Wire development cycle recorder"
    assert draft["domain"] == "development"
    assert draft["affected_resources"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert draft["validation"]["tests"] == ["development cycle tests passed"]
    assert "Ran command: poetry run pytest" in draft["actions"][1]
    assert draft["files"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert draft["meta"]["recorded_from"] == "development_cycle"
    assert draft["meta"]["draft"] is True
    assert handlers["omni_memory_stats"]()["experiences"] == 0


def test_mcp_development_cycle_record_writes_searchable_experience():
    handlers = build_mcp_handlers(_memory())

    written = handlers["omni_memory_record_development_cycle"](
        goal="Wire development cycle recorder",
        summary="Added controlled dev-cycle recording.",
        changed_files=["app/development_cycle.py", "app/integrations/mcp.py"],
        commands_run=["poetry run pytest -q"],
        tests=["83 passed"],
        decisions=["Use explicit lesson before recording"],
        outcome="Development cycle recorder is available through MCP.",
        lesson="Development cycles should be drafted first, then recorded as experience after a reusable lesson is explicit.",
        reuse_when=["recording meaningful development outcomes"],
        source="test",
    )

    assert written["saved"] == 1
    assert handlers["omni_memory_stats"]()["experiences"] == 1

    found = handlers["omni_memory_search_experiences"](
        query="development cycles drafted recorded reusable lesson",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Wire development cycle recorder"
    assert found[0]["refs"]["files"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert found[0]["refs"]["affected_resources"] == ["app/development_cycle.py", "app/integrations/mcp.py"]
    assert "poetry run pytest -q" in found[0]["actions"][1]
    assert found[0]["evaluation"]["tests"] == ["83 passed"]
    assert found[0]["evaluation"]["validation"]["commands_run"] == ["poetry run pytest -q"]
    assert found[0]["meta"]["recorded_from"] == "development_cycle"
    assert found[0]["meta"]["domain"] == "development"


def test_mcp_finish_development_task_records_experience_and_returns_distillation_review():
    memory = _memory(distiller=FakeWorkflowDistiller())
    handlers = build_mcp_handlers(memory)

    result = handlers["omni_memory_finish_development_task"](
        goal="Make development memory workflow automatic",
        summary="Added finish task workflow.",
        changed_files=["app/development_memory_workflow.py", "app/integrations/mcp.py"],
        commands_run=["poetry run pytest -q tests/test_development_cycle.py"],
        tests=["development workflow tests passed"],
        decisions=["Distillation stays dry-run by default"],
        outcome="MCP tool can finish a development task.",
        lesson="Finish-task workflow should record experience and return distillation candidates for review without auto-writing facts.",
        reuse_when=["ending a meaningful coding task"],
        session_turns=[
            {
                "role": "assistant",
                "content": "OmniMemory exposes a finish development task MCP tool.",
            }
        ],
        source="test",
    )

    assert result["experience"]["saved"] == 1
    assert result["distillation"]["saved"][0]["predicate"] == "exposes"
    assert result["distillation"]["operations"][0]["meta"]["dry_run"] is True
    assert result["review_candidates"][0]["object"] == "finish development task MCP tool"
    assert "distillation_dry_run" in result["advisories"]
    assert handlers["omni_memory_stats"]()["experiences"] == 1
    assert handlers["omni_memory_stats"]()["facts"] == 0

    found = handlers["omni_memory_search_experiences"](
        query="finish task workflow distillation candidates review",
        k=3,
    )["experiences"]
    assert found[0]["goal"] == "Make development memory workflow automatic"
    assert found[0]["meta"]["recorded_from"] == "development_memory_workflow"
    assert found[0]["meta"]["domain"] == "development"
    assert found[0]["refs"]["affected_resources"] == ["app/development_memory_workflow.py", "app/integrations/mcp.py"]


def test_mcp_schemas_and_fastmcp_include_development_cycle_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}
    assert "omni_memory_draft_development_cycle" in names
    assert "omni_memory_record_development_cycle" in names
    assert "omni_memory_finish_development_task" in names

    schema = next(tool for tool in MCP_TOOL_SCHEMAS if tool["name"] == "omni_memory_finish_development_task")
    assert schema["inputSchema"]["required"] == ["goal", "lesson"]
    assert "session_turns" in schema["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_fastmcp_lists_development_cycle_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_draft_development_cycle" for tool in listed)
    assert any(tool.name == "omni_memory_record_development_cycle" for tool in listed)
    assert any(tool.name == "omni_memory_finish_development_task" for tool in listed)

    called = await server.call_tool(
        "omni_memory_draft_development_cycle",
        {"goal": "Check FastMCP dev-cycle tool"},
    )
    body = _tool_text(called)
    assert body["goal"] == "Check FastMCP dev-cycle tool"

    finished = await server.call_tool(
        "omni_memory_finish_development_task",
        {
            "goal": "Check FastMCP finish task tool",
            "lesson": "Finish task tool can record experience without running distillation.",
            "run_distiller": False,
        },
    )
    finished_body = _tool_text(finished)
    assert finished_body["experience"]["saved"] == 1
    assert finished_body["distillation"] is None
