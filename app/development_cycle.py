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

    def to_agent_cycle(self) -> AgentCycleRecord:
        validation = {
            "tests": self.tests,
            "commands_run": self.commands_run,
            "ci_status": self.meta.get("ci_status", "unknown"),
        }
        evaluation = {
            "tests": self.tests,
            "side_effects": self.side_effects,
            "validation": validation,
        }
        refs = {
            "files": self.changed_files,
            "commands_run": self.commands_run,
        }
        return AgentCycleRecord(
            goal=self.goal,
            context=self.summary,
            plan=[],
            decisions=self.decisions,
            actions=_actions(self),
            outcome=self.outcome or _default_outcome(self),
            evaluation=evaluation,
            lesson=self.lesson or "Review this development cycle and fill a reusable lesson before recording.",
            confidence=self.confidence,
            refs=refs,
            affected_resources=self.changed_files,
            validation=validation,
            domain="development",
            meta={
                "recorded_from": "development_cycle",
                "draft": not bool(self.lesson),
                "domain": "development",
                **self.meta,
            },
            # Legacy compatibility fields consumed by older tests/repositories.
            tests=self.tests,
            files=self.changed_files,
            side_effects=self.side_effects,
            reuse_when=self.reuse_when,
            avoid_when=self.avoid_when,
        )


class DevelopmentCycleRecorder:
    def draft(self, cycle: DevelopmentCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        parsed = cycle if isinstance(cycle, DevelopmentCycleDraft) else DevelopmentCycleDraft.model_validate(cycle)
        return parsed.to_agent_cycle()


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
