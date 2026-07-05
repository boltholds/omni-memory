from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class Provenance(BaseModel):
    source: str = "user"
    time: Optional[float] = None  # epoch seconds
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryScope(BaseModel):
    tenant_id: str = "default"
    agent_id: str | None = None
    domain_ids: List[str] = Field(default_factory=list)
    environment: Literal["prod", "dev", "test", "benchmark", "sandbox"] = "dev"
    durability: Literal["durable", "ephemeral", "session"] = "durable"
    visibility: Literal["private", "shared", "global"] = "private"
    exclude_from_consolidation: bool = False


class DomainNode(BaseModel):
    id: str
    name: str
    kind: Literal[
        "project",
        "subdomain",
        "knowledge_area",
        "environment",
        "artifact_group",
        "team",
        "product",
    ] = "knowledge_area"
    aliases: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DomainLink(BaseModel):
    source_id: str
    relation: Literal[
        "belongs_to",
        "has_subdomain",
        "related_to",
        "depends_on",
        "shared_with",
        "applies_to",
        "derived_from",
        "verified_by",
        "supersedes",
    ]
    target_id: str
    confidence: float = 1.0
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryObject(BaseModel):
    id: str
    type: str
    payload: Dict[str, Any]
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class FactBelief(BaseModel):
    key: str
    subject: str
    predicate: str
    current: Optional[Fact] = None
    alternatives: List[Fact] = Field(default_factory=list)
    historical: List[Fact] = Field(default_factory=list)
    variants: List[str] = Field(default_factory=list)
    current_score: float = 0.0
    status: str = "unknown"
    reason: str = ""


class Episode(BaseModel):
    id: str
    participants: List[str] = Field(default_factory=list)
    summary: str = ""
    events: List[EpisodeEvent] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)

class EpisodeEvent(BaseModel):
    t: Optional[float] = None
    event_type: str = "note"
    summary: str
    refs: Dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    id: str
    title: str
    status: str = "accepted"
    context: str = ""
    decision: str = ""
    consequences: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    refs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExperienceRecord(BaseModel):
    id: str
    goal: str
    context: str = ""
    decision: str = ""
    actions: List[str] = Field(default_factory=list)
    outcome: str = ""
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    lesson: str = ""
    reuse_when: List[str] = Field(default_factory=list)
    avoid_when: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    refs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SkillRecord(BaseModel):
    id: str
    name: str
    problem: str = ""
    procedure: List[str] = Field(default_factory=list)
    reuse_when: List[str] = Field(default_factory=list)
    avoid_when: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    refs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class FailurePatternRecord(BaseModel):
    id: str
    symptom: str
    root_cause: str = ""
    fix: str = ""
    detection: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    refs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ReviewItem(BaseModel):
    id: str
    kind: Literal["decision", "skill", "failure_pattern", "writeback_item"]
    title: str
    payload: Dict[str, Any]
    status: Literal["proposed", "accepted", "rejected", "superseded"] = "proposed"
    confidence: float = 0.5
    reason: str = ""
    superseded_by: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[float] = None
    provenance: Provenance = Field(default_factory=Provenance)
    meta: Dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    semantic_chunks: List[MemoryObject] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    beliefs: List[FactBelief] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    decisions: List[DecisionRecord] = Field(default_factory=list)
    experiences: List[ExperienceRecord] = Field(default_factory=list)
    skills: List[SkillRecord] = Field(default_factory=list)
    failure_patterns: List[FailurePatternRecord] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)

class ConflictItem(BaseModel):
    key: str
    variants: List[str]

class ConflictReport(BaseModel):
    conflicts: List[ConflictItem] = Field(default_factory=list)

class ContextSection(BaseModel):
    title: str
    body: str

class ContextPack(BaseModel):
    sections: List[ContextSection] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)

class WriteReport(BaseModel):
    saved: int = 0
    rejected: int = 0
    reasons: List[str] = Field(default_factory=list)

class QuerySpec(TypedDict, total=False):
    subject: str
    predicate: str
    object: str
