from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from .models import MemoryObject, Fact, Episode, DecisionRecord, RetrievalBundle, ConflictReport, ContextPack

class IMemoryReadRepository(ABC):
    @abstractmethod
    def semantic_search(self, text: str, k: int = 5) -> List[MemoryObject]: ...

class IMemoryWriteRepository(ABC):
    @abstractmethod
    def save_object(self, obj: MemoryObject) -> None: ...

class IGraphRepository(ABC):
    @abstractmethod
    def save_fact(self, fact: Fact) -> None: ...
    @abstractmethod
    def query(self, **query_spec) -> List[Fact]: ...

class IEpisodicRepository(ABC):
    @abstractmethod
    def save_episode(self, episode: Episode) -> None: ...
    @abstractmethod
    def search(self, user: str | None, entities: list[str], k: int = 5) -> List[Episode]: ...

class IDecisionRepository(ABC):
    @abstractmethod
    def save_decision(self, decision: DecisionRecord) -> None: ...
    @abstractmethod
    def get_decision(self, decision_id: str) -> DecisionRecord | None: ...
    @abstractmethod
    def list_decisions(self, status: str | None = None, limit: int | None = None) -> List[DecisionRecord]: ...
    @abstractmethod
    def search(self, text: str, k: int = 5) -> List[DecisionRecord]: ...

class IConsistencyEngine(ABC):
    @abstractmethod
    def detect_conflicts(self, facts: List[Fact]) -> ConflictReport: ...

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle: ...

class IMemoryOrchestrator(ABC):
    @abstractmethod
    def plan_retrieval(self, query: str) -> RetrievalBundle: ...
    @abstractmethod
    def assemble_context(self, bundle: RetrievalBundle) -> ContextPack: ...
