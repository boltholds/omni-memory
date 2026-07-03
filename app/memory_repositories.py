from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings
from domain.repositories import IFactRepo
from infra.repo.decision_repo import DecisionRepo
from infra.repo.episodic_repo import EpisodicRepo
from infra.repo.experience_repo import ExperienceRepo
from infra.repo.graph_repo import GraphRepo
from infra.repo.vector_repo import VectorStoreRepo


@dataclass(frozen=True)
class MemoryClearReport:
    vector_objects: int = 0
    facts: int = 0
    episodes: int = 0
    decisions: int = 0
    experiences: int = 0
    session_turns: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class MemoryRepositoryCounts:
    vector_objects: int | None = None
    facts: int | None = None
    episodes: int | None = None
    decisions: int | None = None
    experiences: int | None = None


@dataclass(frozen=True)
class MemoryClearCommand:
    include_vectors: bool = True
    include_facts: bool = True
    include_episodes: bool = True
    include_decisions: bool = True
    include_experiences: bool = True
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

    def count(self) -> MemoryRepositoryCounts:
        return MemoryRepositoryCounts(
            vector_objects=_repo_count(self.vector),
            facts=_repo_count(self.graph),
            episodes=_repo_count(self.episodic),
            decisions=_repo_count(self.decision),
            experiences=_repo_count(self.experience),
        )

    def clear(
        self,
        *,
        include_vectors: bool = True,
        include_facts: bool = True,
        include_episodes: bool = True,
        include_decisions: bool = True,
        include_experiences: bool = True,
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
) -> MemoryRepositories:
    return MemoryRepositories(
        vector=vector_repo or VectorStoreRepo(embedder=embedder),
        graph=graph_repo or GraphRepo(),
        episodic=episodic_repo or EpisodicRepo(db_path=settings.sqlite_path),
        decision=decision_repo or DecisionRepo(),
        experience=experience_repo or ExperienceRepo(),
    )


def _repo_count(repo: Any) -> int | None:
    return int(repo.count()) if hasattr(repo, "count") else None


def _report_count(value: int | None, *, include: bool) -> int:
    if not include or value is None:
        return 0
    return value


def _repo_clear(repo: Any) -> int:
    if not hasattr(repo, "clear"):
        return 0
    return int(repo.clear())
