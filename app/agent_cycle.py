from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentCycleRecord(BaseModel):
    goal: str
    plan: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    outcome: str = ""
    tests: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    lesson: str
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    meta: dict[str, Any] = Field(default_factory=dict)


def experience_from_agent_cycle(record: AgentCycleRecord) -> dict[str, Any]:
    evaluation: dict[str, Any] = {}
    if record.tests:
        evaluation["tests"] = record.tests
    if record.side_effects:
        evaluation["side_effects"] = record.side_effects

    context_parts: list[str] = []
    if record.plan:
        context_parts.append("Plan: " + " | ".join(record.plan))
    if record.decisions:
        context_parts.append("Decisions: " + " | ".join(record.decisions))

    refs: dict[str, Any] = {}
    if record.files:
        refs["files"] = record.files

    return {
        "goal": record.goal,
        "context": "\n".join(context_parts),
        "decision": record.decisions[-1] if record.decisions else "",
        "actions": record.actions,
        "outcome": record.outcome,
        "evaluation": evaluation,
        "lesson": record.lesson,
        "reuse_when": record.reuse_when,
        "avoid_when": record.avoid_when,
        "confidence": record.confidence,
        "refs": refs,
        "meta": record.meta,
    }
