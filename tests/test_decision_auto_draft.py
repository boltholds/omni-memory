from __future__ import annotations

import json
import logging

from omni_memory.builder import build_memory
from omni_memory.decision_auto_draft import draft_decision_candidates
from omni_memory.development_memory_workflow import FinishDevelopmentTaskRequest
from omni_memory.integrations.mcp import build_mcp_handlers
from omni_memory.infra.embeddings.factory import HashEmbedder


class FakeDecisionLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages = []

    def generate(self, messages, temperature: float = 0.3):
        self.messages.append({"messages": messages, "temperature": temperature})
        return {"text": self.text, "model": "fake-decision-llm"}


def _memory(*, llm=None):
    return build_memory(use_llm=False, embedder=HashEmbedder(), llm=llm)


def _registry_refactor_request() -> FinishDevelopmentTaskRequest:
    return FinishDevelopmentTaskRequest(
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


def test_decision_auto_draft_detects_mcp_registry_refactor():
    candidates = draft_decision_candidates(_registry_refactor_request())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "proposed"
    assert candidate.meta["auto_draft"] is True
    assert candidate.meta["review_required"] is True
    assert candidate.meta["drafted_by"] == "heuristic"
    assert "Centralize MCP schema" in candidate.title
    assert "required fields" in candidate.decision
    assert candidate.refs["files"] == ["app/integrations/mcp.py", "app/mcp_server.py"]
    assert candidate.refs["tests"] == ["MCP server tests passed"]


def test_model_assisted_decision_auto_draft_uses_llm_candidate_first():
    llm = FakeDecisionLLM(
        json.dumps(
            {
                "decision_needed": True,
                "candidates": [
                    {
                        "title": "Adopt shared MCP schema registry",
                        "decision": "Use one shared registry as the source of truth for MCP tool schemas.",
                        "context": "Handlers and FastMCP tools must stay aligned.",
                        "consequences": ["Schema changes now happen in one place."],
                        "alternatives": ["Keep duplicate schema definitions."],
                        "confidence": 0.88,
                        "reason": "The task changed API/tool schema architecture.",
                    }
                ],
            }
        )
    )

    candidates = draft_decision_candidates(_registry_refactor_request(), llm=llm)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.title == "Adopt shared MCP schema registry"
    assert candidate.decision == "Use one shared registry as the source of truth for MCP tool schemas."
    assert candidate.status == "proposed"
    assert candidate.meta["drafted_by"] == "model"
    assert candidate.meta["review_required"] is True
    assert candidate.refs["files"] == ["app/integrations/mcp.py", "app/mcp_server.py"]
    assert llm.messages[0]["temperature"] == 0.0
    assert "Development task JSON" in llm.messages[0]["messages"][1]["content"]


def test_model_can_suppress_heuristic_candidate_when_decision_not_needed():
    llm = FakeDecisionLLM('{"decision_needed": false, "candidates": []}')

    assert draft_decision_candidates(_registry_refactor_request(), llm=llm) == []


def test_invalid_model_output_falls_back_to_heuristics(caplog):
    llm = FakeDecisionLLM("not json")
    caplog.set_level(logging.WARNING, logger="app.decision_auto_draft")

    candidates = draft_decision_candidates(_registry_refactor_request(), llm=llm)

    assert len(candidates) == 1
    assert candidates[0].meta["drafted_by"] == "heuristic"
    assert any(record.message == "decision_auto_draft_model_failed" for record in caplog.records)
    warning = next(record for record in caplog.records if record.message == "decision_auto_draft_model_failed")
    assert warning.op == "draft_with_model"
    assert warning.fallback == "heuristics"


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


def test_finish_development_task_returns_model_decision_candidate_without_auto_writing_decision():
    llm = FakeDecisionLLM(
        json.dumps(
            {
                "decision_needed": True,
                "candidates": [
                    {
                        "title": "Adopt shared MCP schema registry",
                        "decision": "Use one shared registry as the source of truth for MCP tool schemas.",
                        "context": "Handlers and FastMCP tools must stay aligned.",
                        "consequences": ["Schema changes now happen in one place."],
                        "alternatives": [],
                        "confidence": 0.86,
                        "reason": "The task changed API/tool schema architecture.",
                    }
                ],
            }
        )
    )
    handlers = build_mcp_handlers(_memory(llm=llm))

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
    assert result["decision_candidates"][0]["title"] == "Adopt shared MCP schema registry"
    assert result["decision_candidates"][0]["meta"]["drafted_by"] == "model"
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
