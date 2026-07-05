from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
from omni_memory.domain.models import (
    ContextPack,
    ConflictReport,
    DecisionRecord,
    Episode,
    ExperienceRecord,
    Fact,
    FailurePatternRecord,
    MemoryObject,
    RetrievalBundle,
    ReviewItem,
    SkillRecord,
)

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

class IExperienceRepository(ABC):
    @abstractmethod
    def save_experience(self, experience: ExperienceRecord) -> None: ...
    @abstractmethod
    def get_experience(self, experience_id: str) -> ExperienceRecord | None: ...
    @abstractmethod
    def list_experiences(self, limit: int | None = None) -> List[ExperienceRecord]: ...
    @abstractmethod
    def search(self, text: str, k: int = 5) -> List[ExperienceRecord]: ...

class ISkillRepository(ABC):
    @abstractmethod
    def save_skill(self, skill: SkillRecord) -> None: ...
    @abstractmethod
    def get_skill(self, skill_id: str) -> SkillRecord | None: ...
    @abstractmethod
    def list_skills(self, limit: int | None = None) -> List[SkillRecord]: ...
    @abstractmethod
    def search(self, text: str, k: int = 5) -> List[SkillRecord]: ...

class IFailurePatternRepository(ABC):
    @abstractmethod
    def save_failure_pattern(self, pattern: FailurePatternRecord) -> None: ...
    @abstractmethod
    def get_failure_pattern(self, pattern_id: str) -> FailurePatternRecord | None: ...
    @abstractmethod
    def list_failure_patterns(self, limit: int | None = None) -> List[FailurePatternRecord]: ...
    @abstractmethod
    def search(self, text: str, k: int = 5) -> List[FailurePatternRecord]: ...

class IReviewQueueRepository(ABC):
    @abstractmethod
    def save_review_item(self, item: ReviewItem) -> None: ...
    @abstractmethod
    def get_review_item(self, item_id: str) -> ReviewItem | None: ...
    @abstractmethod
    def list_review_items(self, status: str | None = None, kind: str | None = None, limit: int | None = None) -> List[ReviewItem]: ...
    @abstractmethod
    def count(self) -> int: ...

class IConsistencyEngine(ABC):
    @abstractmethod
    def detect_conflicts(self, facts: List[Fact]) -> ConflictReport: ...

class IRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k_sem: int = 5,
        k_eps: int = 3,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle: ...

class IMemoryOrchestrator(ABC):
    @abstractmethod
    def plan_retrieval(
        self,
        query: str,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle: ...
    @abstractmethod
    def assemble_context(
        self,
        bundle: RetrievalBundle,
        intent: str | None = None,
        mode: str | None = None,
    ) -> ContextPack: ...
