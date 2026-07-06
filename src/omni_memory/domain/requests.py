from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WriteDecisionRequest:
    title: str
    decision: str
    context: str = ""
    consequences: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    refs: dict[str, Any] = field(default_factory=dict)
    status: str = "accepted"
    source: str = "user"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordExperienceRequest:
    goal: str
    lesson: str
    context: str = ""
    decision: str = ""
    actions: list[str] = field(default_factory=list)
    outcome: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    reuse_when: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    confidence: float = 0.5
    refs: dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteSkillRequest:
    name: str
    problem: str = ""
    procedure: list[str] = field(default_factory=list)
    reuse_when: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    refs: dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteFailurePatternRequest:
    symptom: str
    root_cause: str = ""
    fix: str = ""
    detection: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    refs: dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    meta: dict[str, Any] = field(default_factory=dict)
