from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

class Provenance(BaseModel):
    source: str = "user"
    time: Optional[float] = None  # epoch seconds
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


class RetrievalBundle(BaseModel):
    semantic_chunks: List[MemoryObject] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    beliefs: List[FactBelief] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    decisions: List[DecisionRecord] = Field(default_factory=list)
    experiences: List[ExperienceRecord] = Field(default_factory=list)
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
