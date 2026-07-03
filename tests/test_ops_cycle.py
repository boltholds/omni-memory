from __future__ import annotations

import json

import pytest

from app.builder import build_memory
from app.integrations.mcp import MCP_TOOL_SCHEMAS, build_mcp_handlers
from app.mcp_server import build_mcp_app
from app.ops_cycle import OpsCycleDraft
from domain.experience_evaluator import DomainExperienceEvaluator
from domain.models import ExperienceRecord
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _tool_text(response) -> dict:
    return json.loads(response[0].text)


def test_ops_cycle_draft_adapts_to_domain_neutral_agent_cycle():
    cycle = OpsCycleDraft(
        goal="Restore API latency SLA",
        service="api",
        alert_id="alert-123",
        symptoms=["p95 latency spike after deploy"],
        actions=["Inspect traces", "Rollback cache config"],
        outcome="Latency returned to baseline",
        metrics_before={"latency_p95_ms": 2300, "error_rate": 0.08},
        metrics_after={"latency_p95_ms": 280, "error_rate": 0.01},
        lesson="Rollback cache configuration when p95 latency spikes after deploy.",
        affected_resources=["deployment:cache-config"],
        meta={"sla_restored": True},
    ).to_agent_cycle()

    assert cycle.domain == "ops"
    assert cycle.meta["domain"] == "ops"
    assert cycle.meta["service"] == "api"
    assert cycle.affected_resources == ["service:api", "deployment:cache-config", "alert:alert-123"]
    assert cycle.validation["sla_restored"] is True
    assert cycle.validation["metrics_before"]["latency_p95_ms"] == 2300
    assert cycle.validation["metrics_after"]["latency_p95_ms"] == 280
    assert cycle.refs["service"] == "api"
    assert cycle.refs["alert_id"] == "alert-123"


def test_mcp_ops_cycle_record_writes_searchable_experience():
    handlers = build_mcp_handlers(_memory())

    written = handlers["omni_memory_record_ops_cycle"](
        goal="Restore API latency SLA",
        service="api",
        alert_id="alert-123",
        symptoms=["p95 latency spike after deploy"],
        actions=["Inspect traces", "Rollback cache config"],
        outcome="Latency returned to baseline",
        metrics_before={"latency_p95_ms": 2300, "error_rate": 0.08},
        metrics_after={"latency_p95_ms": 280, "error_rate": 0.01},
        lesson="Rollback cache configuration when p95 latency spikes after deploy.",
        reuse_when=["p95 latency spike after deploy"],
        source="codex-dev",
        meta={"domain_ids": ["domain:ops:api"], "sla_restored": True},
    )

    assert written["saved"] == 1
    found = handlers["omni_memory_search_experiences"](
        query="latency SLA cache rollback",
        k=3,
    )["experiences"]

    assert found[0]["goal"] == "Restore API latency SLA"
    assert found[0]["meta"]["domain"] == "ops"
    assert found[0]["meta"]["service"] == "api"
    assert found[0]["evaluation"]["validation"]["sla_restored"] is True
    assert found[0]["refs"]["affected_resources"][0] == "service:api"


def test_ops_experience_evaluator_is_registered_by_default():
    evaluator = DomainExperienceEvaluator()
    experience = ExperienceRecord(
        id="exp-ops",
        goal="Restore API latency SLA",
        context="Service: api\nSymptoms: p95 latency spike after deploy",
        actions=["Inspect traces", "Rollback cache config"],
        outcome="Latency returned to baseline",
        evaluation={
            "validation": {
                "service": "api",
                "sla_restored": True,
                "metrics_before": {"latency_p95_ms": 2300},
                "metrics_after": {"latency_p95_ms": 280},
            }
        },
        lesson="Rollback cache configuration when p95 latency spikes after deploy.",
        reuse_when=["p95 latency spike after deploy"],
        confidence=0.91,
        meta={"domain": "ops"},
    )

    result = evaluator.evaluate(experience)

    assert result.meta["routed_domain"] == "ops"
    assert result.meta["evaluator"] == "OpsExperienceEvaluator"
    assert result.recommended_memory_type in {"skill", "both"}
    assert result.success_score >= 0.6
    assert "ops" in result.consolidation_tags


def test_ops_consolidation_proposes_skill_from_repeated_incidents():
    memory = _memory()

    for suffix in ["one", "two"]:
        memory.record_ops_cycle(
            {
                "goal": "Restore API latency SLA",
                "service": "api",
                "alert_id": f"alert-{suffix}",
                "symptoms": ["p95 latency spike after deploy"],
                "actions": ["Inspect traces", "Rollback cache config"],
                "outcome": f"Latency returned to baseline {suffix}",
                "metrics_before": {"latency_p95_ms": 2300, "error_rate": 0.08},
                "metrics_after": {"latency_p95_ms": 280, "error_rate": 0.01},
                "lesson": "Rollback cache configuration when p95 latency spikes after deploy.",
                "reuse_when": ["p95 latency spike after deploy"],
                "confidence": 0.91,
                "meta": {"domain_ids": ["domain:ops:api"], "sla_restored": True},
            },
            source="codex-dev",
        )

    result = memory.consolidate_experiences(dry_run=True, min_confidence=0.85)

    assert any(proposal.kind == "skill" for proposal in result.proposals)
    skill = next(proposal for proposal in result.proposals if proposal.kind == "skill")
    assert skill.payload["meta"]["domain"] == "ops"
    assert skill.payload["meta"]["scope"]["domain_ids"] == ["domain:ops:api"]


def test_mcp_schemas_and_fastmcp_include_ops_cycle_tools():
    names = {tool["name"] for tool in MCP_TOOL_SCHEMAS}
    assert "omni_memory_draft_ops_cycle" in names
    assert "omni_memory_record_ops_cycle" in names

    draft_schema = next(tool for tool in MCP_TOOL_SCHEMAS if tool["name"] == "omni_memory_draft_ops_cycle")
    record_schema = next(tool for tool in MCP_TOOL_SCHEMAS if tool["name"] == "omni_memory_record_ops_cycle")
    assert draft_schema["inputSchema"]["required"] == ["goal", "service"]
    assert record_schema["inputSchema"]["required"] == ["goal", "service", "lesson"]
    assert "metrics_before" in record_schema["inputSchema"]["properties"]
    assert "metrics_after" in record_schema["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_fastmcp_lists_and_calls_ops_cycle_tools():
    server = build_mcp_app(_memory())

    listed = await server.list_tools()
    assert any(tool.name == "omni_memory_draft_ops_cycle" for tool in listed)
    assert any(tool.name == "omni_memory_record_ops_cycle" for tool in listed)

    called = await server.call_tool(
        "omni_memory_draft_ops_cycle",
        {"goal": "Restore API latency SLA", "service": "api"},
    )
    body = _tool_text(called)
    assert body["domain"] == "ops"
    assert body["refs"]["service"] == "api"
