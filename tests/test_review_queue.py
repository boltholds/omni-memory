from __future__ import annotations

import json

import pytest

from omni_memory import build_memory
from omni_memory.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from omni_memory.mcp_server import build_mcp_app
from omni_memory.infra.embeddings.factory import HashEmbedder
from omni_memory.infra.repo.review_repo import PersistentReviewQueueRepo, ReviewQueueRepo


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_review_queue_accepts_decision_candidate_before_writing_adr():
    handlers = build_mcp_handlers(_memory())

    result = handlers["omni_memory_finish_development_task"](
        goal="Refactor MCP schema registry",
        summary="Centralized MCP tool schema definitions and kept required fields explicit.",
        changed_files=["app/integrations/mcp.py", "app/mcp_server.py"],
        tests=["MCP tests passed"],
        decisions=["Use one registry as the source of truth for MCP tool schemas"],
        outcome="Handlers and FastMCP use aligned schema metadata.",
        lesson="MCP schema changes should be centralized so handlers and FastMCP stay aligned.",
        reuse_when=["changing MCP tools"],
        source="test",
        run_distiller=False,
    )

    assert result["decision_candidates"]
    review_id = result["decision_candidates"][0]["meta"]["review_item_id"]
    assert "decision_candidates_review" in result["advisories"]
    assert handlers["omni_memory_stats"]()["decisions"] == 0

    queued = handlers["omni_memory_get_review_item"](item_id=review_id)["review_item"]
    assert queued["status"] == "proposed"
    assert queued["kind"] == "decision"
    assert queued["payload"]["status"] == "proposed"

    accepted = handlers["omni_memory_accept_review_item"](
        item_id=review_id,
        reviewer="test",
        note="ADR accepted",
    )

    assert accepted["applied"] is True
    assert accepted["item"]["status"] == "accepted"
    assert accepted["result"]["saved"] == 1
    decisions = handlers["omni_memory_list_decisions"](status="accepted")["decisions"]
    assert decisions[0]["status"] == "accepted"
    assert decisions[0]["meta"]["accepted_from_review_id"] == review_id


def test_review_queue_reject_keeps_candidate_out_of_memory():
    handlers = build_mcp_handlers(_memory())

    submitted = handlers["omni_memory_submit_review_item"](
        kind="skill",
        title="Use behavior tests",
        payload={
            "name": "Use behavior tests",
            "procedure": ["Test observable behavior"],
            "reuse_when": ["adding tests"],
            "confidence": 0.9,
        },
        confidence=0.9,
        source="test",
    )

    rejected = handlers["omni_memory_reject_review_item"](
        item_id=submitted["id"],
        reviewer="test",
        note="Too generic",
    )

    assert rejected["applied"] is True
    assert rejected["item"]["status"] == "rejected"
    assert handlers["omni_memory_stats"]()["skills"] == 0
    assert handlers["omni_memory_list_review_items"](status="rejected")["review_items"][0]["id"] == submitted["id"]


def test_review_queue_supersede_creates_replacement_without_applying_old_item():
    handlers = build_mcp_handlers(_memory())
    submitted = handlers["omni_memory_submit_review_item"](
        kind="skill",
        title="Old skill",
        payload={"name": "Old skill"},
        source="test",
    )

    superseded = handlers["omni_memory_supersede_review_item"](
        item_id=submitted["id"],
        replacement={
            "kind": "skill",
            "title": "Better skill",
            "payload": {"name": "Better skill", "procedure": ["Do the better thing"]},
        },
        reviewer="test",
        note="Improve wording",
    )

    assert superseded["item"]["status"] == "superseded"
    assert superseded["created"]["status"] == "proposed"
    assert superseded["item"]["superseded_by"] == superseded["created"]["id"]
    assert handlers["omni_memory_stats"]()["skills"] == 0


def test_persistent_review_queue_repo_reloads_and_clears(tmp_path):
    path = tmp_path / "review_queue.json"
    repo = PersistentReviewQueueRepo(ReviewQueueRepo(), path)
    repo.save_review_item(
        _memory().submit_review_item(
            kind="decision",
            title="Adopt shared registry",
            payload={"title": "Adopt shared registry", "decision": "Use one registry."},
            source="test",
        )
    )

    reloaded = PersistentReviewQueueRepo(ReviewQueueRepo(), path)

    assert reloaded.count() == 1
    assert reloaded.list_review_items(kind="decision")[0].title == "Adopt shared registry"
    assert reloaded.clear() == 1
    assert PersistentReviewQueueRepo(ReviewQueueRepo(), path).count() == 0


def test_mcp_review_tools_are_advertised_and_callable():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}
    handlers = build_mcp_handlers(_memory())

    assert "omni_memory_submit_review_item" in names
    assert "omni_memory_accept_review_item" in names
    assert "omni_memory_submit_review_item" in handlers


@pytest.mark.asyncio
async def test_mcp_server_lists_review_queue_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_list_review_items" for tool in listed)

    called = await server.call_tool("omni_memory_submit_review_item", {
        "kind": "decision",
        "title": "Adopt review queue",
        "payload": {"title": "Adopt review queue", "decision": "Review candidates before accepting them."},
    })
    body = _tool_text(called)
    assert body["status"] == "proposed"
