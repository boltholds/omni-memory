from __future__ import annotations

from omni_memory import build_memory
from omni_memory.domain.models import RetrievalBundle
from omni_memory.infra.embeddings.factory import HashEmbedder


TASK = "Add a new MCP tool for exporting memory summaries without breaking advertised tool discovery."


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def _agent_plan(task: str, memory: RetrievalBundle | None = None) -> list[str]:
    plan = [
        "Add the MCP handler for the requested export operation.",
        "Run the focused MCP server tests.",
    ]

    patterns = memory.failure_patterns if memory is not None else []
    if any("registry" in pattern.fix.lower() for pattern in patterns):
        plan.insert(0, "Update the declarative MCP schema registry before wiring the handler.")
        plan.append("Assert advertised MCP tools and runtime handlers stay in sync.")

    return plan


def _fake_mcp_contract_check(plan: list[str]) -> bool:
    text = " ".join(plan).lower()
    return "registry" in text and "advertised mcp tools" in text and "handler" in text


def test_coding_agent_demo_reuses_failure_pattern_to_avoid_repeating_a_mcp_bug():
    memory = _memory()

    baseline_plan = _agent_plan(TASK)
    assert _fake_mcp_contract_check(baseline_plan) is False

    memory.write_failure_pattern(
        symptom="A new MCP tool handler was added, but the tool was missing from advertised discovery.",
        root_cause="The runtime handler and declarative MCP schema registry were changed separately.",
        fix="Update the shared MCP schema registry and the runtime handler together, then assert advertised MCP tools match handlers.",
        detection="MCP client cannot discover the new tool, or the advertised-tools contract test fails.",
        confidence=0.95,
        source="demo",
        meta={"domain_ids": ["domain:project:omni-memory"], "demo": "coding-agent-memory-loop"},
    )

    recalled = memory.retrieve(TASK, intent="debug_failure", k_sem=0, k_eps=5)
    assert recalled.failure_patterns
    assert any("schema registry" in pattern.fix for pattern in recalled.failure_patterns)

    memory_aware_plan = _agent_plan(TASK, recalled)
    assert _fake_mcp_contract_check(memory_aware_plan) is True
