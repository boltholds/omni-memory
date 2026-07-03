from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentCycleRecord(BaseModel):
    """Domain-neutral record of one meaningful agent work cycle.

    `tests`, `files` and `side_effects` are retained as legacy/development
    compatibility fields. New domain adapters should prefer `domain`,
    `affected_resources`, `validation`, `evaluation` and `refs`.
    """

    goal: str
    context: str = ""
    plan: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    outcome: str = ""
    evaluation: dict[str, Any] = Field(default_factory=dict)
    lesson: str
    confidence: float = 0.8
    refs: dict[str, Any] = Field(default_factory=dict)
    affected_resources: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    domain: str = "generic"
    meta: dict[str, Any] = Field(default_factory=dict)

    # Legacy/dev-specific compatibility fields.
    tests: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)


def experience_from_agent_cycle(record: AgentCycleRecord) -> dict[str, Any]:
    evaluation = dict(record.evaluation)
    if record.tests:
        evaluation.setdefault("tests", record.tests)
    if record.side_effects:
        evaluation.setdefault("side_effects", record.side_effects)
    if record.validation:
        evaluation.setdefault("validation", record.validation)

    context_parts: list[str] = []
    if record.context:
        context_parts.append(record.context)
    if record.plan:
        context_parts.append("Plan: " + " | ".join(record.plan))
    if record.decisions:
        context_parts.append("Decisions: " + " | ".join(record.decisions))

    refs = dict(record.refs)
    if record.files:
        refs.setdefault("files", record.files)
    if record.affected_resources:
        refs.setdefault("affected_resources", record.affected_resources)

    meta = dict(record.meta)
    meta.setdefault("domain", record.domain)
    if record.affected_resources:
        meta.setdefault("affected_resources", record.affected_resources)
    if record.validation:
        meta.setdefault("validation", record.validation)

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
        "meta": meta,
    }
