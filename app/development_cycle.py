from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent_cycle import AgentCycleRecord


class DevelopmentCycleDraft(BaseModel):
    goal: str
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    outcome: str = ""
    lesson: str = ""
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    meta: dict[str, Any] = Field(default_factory=dict)


class DevelopmentCycleRecorder:
    def draft(self, cycle: DevelopmentCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        parsed = (
            cycle
            if isinstance(cycle, DevelopmentCycleDraft)
            else DevelopmentCycleDraft.model_validate(cycle)
        )
        return AgentCycleRecord(
            goal=parsed.goal,
            plan=[],
            decisions=parsed.decisions,
            actions=_actions(parsed),
            outcome=parsed.outcome or _default_outcome(parsed),
            tests=parsed.tests,
            files=parsed.changed_files,
            side_effects=parsed.side_effects,
            lesson=parsed.lesson or "Review this development cycle and fill a reusable lesson before recording.",
            reuse_when=parsed.reuse_when,
            avoid_when=parsed.avoid_when,
            confidence=parsed.confidence,
            meta={
                **parsed.meta,
                "recorded_from": "development_cycle",
                "draft": not bool(parsed.lesson),
            },
        )


def _actions(cycle: DevelopmentCycleDraft) -> list[str]:
    actions: list[str] = []
    if cycle.summary:
        actions.append(cycle.summary)
    actions.extend(f"Ran command: {command}" for command in cycle.commands_run)
    if cycle.changed_files:
        actions.append("Changed files: " + ", ".join(cycle.changed_files))
    return actions


def _default_outcome(cycle: DevelopmentCycleDraft) -> str:
    if cycle.tests:
        return "Tests/verification: " + "; ".join(cycle.tests)
    if cycle.commands_run:
        return "Commands completed: " + "; ".join(cycle.commands_run)
    return ""
