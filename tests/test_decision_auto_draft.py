from __future__ import annotations

from app.builder import build_memory
from app.decision_auto_draft import draft_decision_candidates
from app.development_memory_workflow import FinishDevelopmentTaskRequest
from app.integrations.mcp import build_mcp_handlers
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_decision_auto_draft_detects_mcp_registry_refactor():
    request = FinishDevelopmentTaskRequest(
        goal="Refactor MCP schema registry",
        summary="Centralized MCP tool schema definitions and kept required fields explicit.",
        changed_files=["app/integrations/mcp.py", "app/mcp_server.py"],
        commands_run=["poetry run pytest tests/test_mcp_server.py -q"],
        tests=["MCP server tests passed"],
        decisions=["Centralize MCP schema definitions and keep required fields explicit"],
        outcome="MCP handlers and FastMCP server use the same schema registry.",
        lesson="MCP schema changes should be centralized so handlers and FastMCP stay aligned.",
        reuse_when=["changing MCP tools"],
        run_distiller=False,
    )

    candidates = draft_decision_candidates(request)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "proposed"
    assert candidate.meta["auto_draft"] is True
    assert candidate.meta["review_required"] is True
    assert "Centralize MCP schema" in candidate.title
    assert "required fields" in candidate.decision
    assert candidate.refs["files"] == ["app/integrations/mcp.py", "app/mcp_server.py"]
    assert candidate.refs["tests"] == ["MCP server tests passed"]


def test_decision_auto_draft_ignores_small_non_architectural_bugfix():
    request = FinishDevelopmentTaskRequest(
        goal="Fix typo in README",
        summary="Corrected one spelling mistake.",
        changed_files=["README.md"],
        tests=[],
        decisions=[],
        outcome="Typo fixed.",
        lesson="Proofread README changes before committing.",
        run_distiller=False,
    )

    assert draft_decision_candidates(request) == []


def test_finish_development_task_returns_decision_candidate_without_auto_writing_decision():
    handlers = build_mcp_handlers(_memory())

    result = handlers["omni_memory_finish_development_task"](
        goal="Refactor MCP schema registry",
        summary="Centralized MCP tool schema definitions and kept required fields explicit.",
        changed_files=["app/integrations/mcp.py", "app/mcp_server.py"],
        commands_run=["poetry run pytest tests/test_mcp_server.py -q"],
        tests=["MCP server tests passed"],
        decisions=["Centralize MCP schema definitions and keep required fields explicit"],
        outcome="MCP handlers and FastMCP server use the same schema registry.",
        lesson="MCP schema changes should be centralized so handlers and FastMCP stay aligned.",
        reuse_when=["changing MCP tools"],
        source="test",
        run_distiller=False,
    )

    assert result["experience"]["saved"] == 1
    assert result["distillation"] is None
    assert result["decision_candidates"][0]["status"] == "proposed"
    assert result["decision_candidates"][0]["meta"]["review_required"] is True
    assert "decision_candidates_review" in result["advisories"]
    assert handlers["omni_memory_stats"]()["decisions"] == 0


def test_finish_development_task_does_not_return_decision_candidate_for_plain_task():
    handlers = build_mcp_handlers(_memory())

    result = handlers["omni_memory_finish_development_task"](
        goal="Fix typo in README",
        summary="Corrected one spelling mistake.",
        changed_files=["README.md"],
        outcome="Typo fixed.",
        lesson="Proofread README changes before committing.",
        source="test",
        run_distiller=False,
    )

    assert result["experience"]["saved"] == 1
    assert result["decision_candidates"] == []
    assert "decision_candidates_review" not in result["advisories"]
    assert handlers["omni_memory_stats"]()["decisions"] == 0
