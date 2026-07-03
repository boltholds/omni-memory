from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from domain.distiller import SessionTurn
from domain.writeback import WritebackResult


class FinishDevelopmentTaskRequest(BaseModel):
    goal: str
    lesson: str
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    outcome: str = ""
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    source: str = "development-workflow"
    meta: dict[str, Any] = Field(default_factory=dict)
    session_turns: list[SessionTurn] = Field(default_factory=list)
    run_distiller: bool = True
    distill_dry_run: bool = True
    min_confidence: float = 0.75
    clear_session: bool = False


class FinishDevelopmentTaskResult(BaseModel):
    experience: dict[str, Any]
    distillation: dict[str, Any] | None = None
    review_candidates: list[dict[str, Any]] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)


class DevelopmentMemoryWorkflow:
    """Agent-friendly wrapper for ending a meaningful development task."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def finish_task(
        self,
        request: FinishDevelopmentTaskRequest | dict[str, Any],
    ) -> FinishDevelopmentTaskResult:
        parsed = (
            request
            if isinstance(request, FinishDevelopmentTaskRequest)
            else FinishDevelopmentTaskRequest.model_validate(request)
        )
        advisories: list[str] = []

        distillation = None
        review_candidates: list[dict[str, Any]] = []
        if parsed.run_distiller:
            distillation_result = self._distill(parsed)
            distillation = distillation_result.model_dump(mode="json")
            review_candidates = _review_candidates(distillation_result)
            if parsed.distill_dry_run:
                advisories.append("distillation_dry_run")

        experience = self.memory.record_development_cycle(
            {
                "goal": parsed.goal,
                "summary": parsed.summary,
                "changed_files": parsed.changed_files,
                "commands_run": parsed.commands_run,
                "tests": parsed.tests,
                "decisions": parsed.decisions,
                "outcome": parsed.outcome,
                "lesson": parsed.lesson,
                "reuse_when": parsed.reuse_when,
                "avoid_when": parsed.avoid_when,
                "side_effects": parsed.side_effects,
                "confidence": parsed.confidence,
                "meta": {
                    **parsed.meta,
                    "recorded_from": "development_memory_workflow",
                },
            },
            source=parsed.source,
        ).model_dump()

        return FinishDevelopmentTaskResult(
            experience=experience,
            distillation=distillation,
            review_candidates=review_candidates,
            advisories=advisories,
        )

    def _distill(self, request: FinishDevelopmentTaskRequest) -> WritebackResult:
        original_turns = list(getattr(self.memory, "_session_turns", []))
        try:
            for turn in request.session_turns:
                self.memory.ingest_turn(turn.role, turn.content)
            return self.memory.commit_session(
                source=request.source,
                dry_run=request.distill_dry_run,
                meta={
                    **request.meta,
                    "scope": {
                        "environment": "dev",
                        "durability": "session" if request.distill_dry_run else "durable",
                    },
                    "distilled_from": "development_memory_workflow",
                },
                min_confidence=request.min_confidence,
                clear=request.clear_session,
            )
        finally:
            if request.session_turns and not request.clear_session:
                self.memory._session_turns = original_turns


def _review_candidates(result: WritebackResult) -> list[dict[str, Any]]:
    items = [*result.saved, *[item.memory_object for item in result.rejected if item.memory_object is not None]]
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump(mode="json"))
    return out
