from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrieveMemoryInput(BaseModel):
    query: str = Field(..., min_length=1, description="Question or task used to retrieve relevant memory.")
    k_sem: int = Field(default=5, ge=0, le=50, description="Maximum semantic note candidates.")
    k_eps: int = Field(default=3, ge=0, le=50, description="Maximum episodic/cognitive candidates per repository.")
    intent: str | None = Field(default=None, description="Memory planner intent, e.g. answer_question, write_code, debug_failure.")
    mode: str | None = Field(default=None, description="Retrieval mode such as fast, quality, or offline.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional tenant/domain/environment scope filter.")


class BuildContextInput(BaseModel):
    query: str = Field(default="", description="Question or task used to build a memory context pack.")
    intent: str | None = Field(default=None, description="Memory planner intent.")
    mode: str | None = Field(default=None, description="Context/retrieval mode.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional tenant/domain/environment scope filter.")


class AskMemoryInput(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer with OmniMemory context.")
    lang: Literal["en", "ru"] = "en"
    style: Literal["concise", "bullets", "detailed", "plain"] = "concise"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    include_context: bool = True
    intent: str | None = None
    mode: str | None = None
    scope: dict[str, Any] | None = None


class WriteMemoryItemsInput(BaseModel):
    items: list[dict[str, Any]] = Field(..., min_length=1, description="Raw OmniMemory writeback items.")
    source: str = "langchain"
    dry_run: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class WriteNoteInput(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "langchain"
    meta: dict[str, Any] = Field(default_factory=dict)


class WriteFactInput(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    source: str = "langchain"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class WriteExperienceInput(BaseModel):
    goal: str = Field(..., min_length=1)
    lesson: str = Field(..., min_length=1)
    context: str = ""
    decision: str = ""
    actions: list[str] = Field(default_factory=list)
    outcome: str = ""
    evaluation: dict[str, Any] = Field(default_factory=dict)
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    refs: dict[str, Any] = Field(default_factory=dict)
    source: str = "langchain"
    meta: dict[str, Any] = Field(default_factory=dict)
