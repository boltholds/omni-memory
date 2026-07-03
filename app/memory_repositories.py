from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings
from domain.repositories import IFactRepo
from infra.repo.cognitive_repo import FailurePatternRepo, SkillRepo
from infra.repo.decision_repo import DecisionRepo
from infra.repo.domain_graph_repo import DomainGraphRepo
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.experience_repo import ExperienceRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.review_repo import ReviewQueueRepo
from infra.repo.vector_repo import VectorStoreRepo


@dataclass(frozen=True)
class MemoryClearReport:
    vector_objects: int = 0
    facts: int = 0
    episodes: int = 0
    decisions: int = 0
    experiences: int = 0
    skills: int = 0
    failure_patterns: int = 0
    review_items: int = 0
    session_turns: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class MemoryRepositoryCounts:
    vector_objects: int | None = None
    facts: int | None = None
    episodes: int | None = None
    decisions: int | None = None
    experiences: int | None = None
    skills: int | None = None
    failure_patterns: int | None = None
    review_items: int | None = None
    domain_nodes: int | None = None
    domain_links: int | None = None


@dataclass(frozen=True)
class MemoryClearCommand:
    include_vectors: bool = True
    include_facts: bool = True
    include_episodes: bool = True
    include_decisions: bool = True
    include_experiences: bool = True
    include_skills: bool = True
    include_failure_patterns: bool = True
    include_review_items: bool = True
    include_session: bool = True
    dry_run: bool = False

    def execute(
        self,
        repositories: MemoryRepositories,
        *,
        session_turns: int = 0,
        clear_session: Any | None = None,
    ) -> MemoryClearReport:
        counts = repositories.count()
        report = MemoryClearReport(
            vector_objects=_report_count(counts.vector_objects, include=self.include_vectors),
            facts=_report_count(counts.facts, include=self.include_facts),
            episodes=_report_count(counts.episodes, include=self.include_episodes),
            decisions=_report_count(counts.decisions, include=self.include_decisions),
            experiences=_report_count(counts.experiences, include=self.include_experiences),
            skills=_report_count(counts.skills, include=self.include_skills),
            failure_patterns=_report_count(
                counts.failure_patterns,
                include=self.include_failure_patterns,
            ),
            review_items=_report_count(counts.review_items, include=self.include_review_items),
            session_turns=session_turns if self.include_session else 0,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            return report

        repositories.clear(
            include_vectors=self.include_vectors,
            include_facts=self.include_facts,
            include_episodes=self.include_episodes,
            include_decisions=self.include_decisions,
            include_experiences=self.include_experiences,
            include_skills=self.include_skills,
            include_failure_patterns=self.include_failure_patterns,
            include_review_items=self.include_review_items,
        )
        if self.include_session and clear_session is not None:
            clear_session()

        return report


@dataclass
class MemoryRepositories:
    vector: Any
    graph: IFactRepo
    episodic: EpisodicRepo
    decision: Any
    experience: Any
    skill: Any
    failure_pattern: Any
    review_queue: Any
    domain_graph: Any

    def count(self) -> MemoryRepositoryCounts:
        return MemoryRepositoryCounts(
            vector_objects=_repo_count(self.vector),
            facts=_repo_count(self.graph),
            episodes=_repo_count(self.episodic),
            decisions=_repo_count(self.decision),
            experiences=_repo_count(self.experience),
            skills=_repo_count(self.skill),
            failure_patterns=_repo_count(self.failure_pattern),
            review_items=_repo_count(self.review_queue),
            domain_nodes=_repo_count(self.domain_graph),
            domain_links=_repo_link_count(self.domain_graph),
        )

    def clear(
        self,
        *,
        include_vectors: bool = True,
        include_facts: bool = True,
        include_episodes: bool = True,
        include_decisions: bool = True,
        include_experiences: bool = True,
        include_skills: bool = True,
        include_failure_patterns: bool = True,
        include_review_items: bool = True,
    ) -> None:
        if include_vectors:
            _repo_clear(self.vector)
        if include_facts:
            _repo_clear(self.graph)
        if include_episodes:
            _repo_clear(self.episodic)
        if include_decisions:
            _repo_clear(self.decision)
        if include_experiences:
            _repo_clear(self.experience)
        if include_skills:
            _repo_clear(self.skill)
        if include_failure_patterns:
            _repo_clear(self.failure_pattern)
        if include_review_items:
            _repo_clear(self.review_queue)

    def stats(self) -> dict[str, int | None]:
        return self.count().__dict__


def build_memory_repositories(
    *,
    embedder: Any | None = None,
    vector_repo: Any | None = None,
    graph_repo: IFactRepo | None = None,
    episodic_repo: EpisodicRepo | None = None,
    decision_repo: Any | None = None,
    experience_repo: Any | None = None,
    skill_repo: Any | None = None,
    failure_pattern_repo: Any | None = None,
    review_queue_repo: Any | None = None,
    domain_graph_repo: Any | None = None,
) -> MemoryRepositories:
    return MemoryRepositories(
        vector=vector_repo or VectorStoreRepo(embedder=embedder),
        graph=graph_repo or GraphRepo(),
        episodic=episodic_repo or EpisodicRepo(db_path=settings.sqlite_path),
        decision=decision_repo or DecisionRepo(),
        experience=experience_repo or ExperienceRepo(),
        skill=skill_repo or SkillRepo(),
        failure_pattern=failure_pattern_repo or FailurePatternRepo(),
        review_queue=review_queue_repo or ReviewQueueRepo(),
        domain_graph=domain_graph_repo or DomainGraphRepo(),
    )


def _repo_count(repo: Any) -> int | None:
    return int(repo.count()) if hasattr(repo, "count") else None


def _repo_link_count(repo: Any) -> int | None:
    return int(repo.link_count()) if hasattr(repo, "link_count") else None


def _report_count(value: int | None, *, include: bool) -> int:
    if not include or value is None:
        return 0
    return value


def _repo_clear(repo: Any) -> int:
    if not hasattr(repo, "clear"):
        return 0
    return int(repo.clear())
