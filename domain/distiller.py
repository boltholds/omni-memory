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


class IMemoryDistiller(ABC):
    @abstractmethod
    def distill(self, text: str) -> DistillationResult:
        raise NotImplementedError