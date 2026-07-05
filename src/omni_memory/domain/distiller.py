from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, Field


class DistilledFact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    volatility: str = "medium"


class DistilledEpisode(BaseModel):
    summary: str
    participants: list[str] = []
    entities: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class DistilledNote(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class DistillationResult(BaseModel):
    facts: list[DistilledFact] = []
    episodes: list[DistilledEpisode] = []
    notes: list[DistilledNote] = []
    rejected: list[str] = []


class SessionTurn(BaseModel):
    role: str
    content: str


class MemoryCandidate(BaseModel):
    """A conservative memory candidate extracted from a dialogue session.

    The candidate is not written directly. It must pass validation first.
    """

    kind: str = Field(description="fact | preference | episode | note | reject")
    should_write: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence_quote: str = ""
    temporal_scope: str = Field(default="unknown", description="current | past | future | unknown")
    payload: dict = Field(default_factory=dict)


class SessionDistillationResult(BaseModel):
    candidates: list[MemoryCandidate] = []
    rejected: list[str] = []


class ISessionMemoryDistiller(Protocol):
    def distill_session(self, turns: list[SessionTurn]) -> SessionDistillationResult:
        ...


class IMemoryDistiller(ABC):
    @abstractmethod
    def distill(self, text: str) -> DistillationResult:
        raise NotImplementedError