from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrieveMemoryInput(BaseModel):
    query: str = Field(..., description="Question or task text used to retrieve memory.")
    intent: str | None = Field(default=None, description="Memory intent such as answer_question, write_code, make_decision or debug_failure.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional OmniMemory retrieval scope.")
    k_sem: int = Field(default=5, ge=0, description="Maximum semantic memory chunks.")
    k_eps: int = Field(default=3, ge=0, description="Maximum episodic/cognitive memory records per type.")


class ContextMemoryInput(BaseModel):
    query: str = Field(..., description="Question or task text used to build an agent context pack.")
    intent: str | None = Field(default=None, description="Memory intent such as answer_question, write_code, make_decision or debug_failure.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional OmniMemory retrieval scope.")


class WriteMemoryInput(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list, description="Writeback items to store through OmniMemory policies.")
    source: str = Field(default="langchain", description="Source label stored in memory provenance.")
    dry_run: bool = Field(default=False, description="Validate and route writes without saving them.")
    meta: dict[str, Any] | None = Field(default=None, description="Optional writeback metadata.")


class FinishDevelopmentTaskInput(BaseModel):
    goal: str = Field(..., description="Development task goal.")
    lesson: str = Field(..., description="Reusable lesson learned from the task.")
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
    source: str = "langchain-development-workflow"
    run_distiller: bool = False
    distill_dry_run: bool = True
    min_confidence: float = 0.75
    clear_session: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class ConsolidateMemoryInput(BaseModel):
    dry_run: bool = Field(default=True, description="Return proposals without saving skills or failure patterns.")
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
