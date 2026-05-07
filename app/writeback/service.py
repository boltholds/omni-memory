from typing import Any, Optional, Protocol, runtime_checkable

from domain.models import Fact, Episode, MemoryObject

from domain.writeback import (
    WritebackRawItem,
    WritebackResult,
    WritebackRequest,
    WritebackDecision,
    MemoryWritePolicy, 
    DomainMemoryObject,
    WritebackContext
)


from domain.ports import IMemoryWriteRepository
from app.writeback.writeback_policies import (
    FactWritebackPolicy,
    EpisodeWritebackPolicy,
    PreferenceWritebackPolicy,
    NoteWritebackPolicy,
    WritebackPolicyResolver
)

from app.writeback.memory_policies import (
    ProvenancePolicy,
    TTLPolicy,
    PiiPolicy,
    ConflictPolicy,
    ConfidencePolicy,
    DedupPolicy
)


class RepositoryNotFound(Exception):
    """No repository for object typ"""
    ...
    
    


@runtime_checkable
class VectorMemoryRepository(Protocol):
    def save_object(self, obj: MemoryObject) -> None:
        ...


@runtime_checkable
class GraphMemoryRepository(Protocol):
    def save_fact(self, fact: Fact) -> None:
        ...


@runtime_checkable
class EpisodicMemoryRepository(Protocol):
    def save_episode(self, episode: Episode) -> None:
        ...


class MemoryRepositoryRouter:
    """
    Роутер сохранения доменных memory objects.

    Он убирает из WriteBackService знание о том,
    какой объект в какой репозиторий сохраняется.
    """

    def __init__(
        self,
        *,
        vector_repo: VectorMemoryRepository,
        graph_repo: GraphMemoryRepository,
        episodic_repo: EpisodicMemoryRepository,
    ) -> None:
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.episodic_repo = episodic_repo

    def save(self, memory_object: DomainMemoryObject) -> None:
        if isinstance(memory_object, Fact):
            self.graph_repo.save_fact(memory_object)
            return

        if isinstance(memory_object, Episode):
            self.episodic_repo.save_episode(memory_object)
            return

        if isinstance(memory_object, MemoryObject):
            self.vector_repo.save_object(memory_object)
            return

        raise TypeError(
            f"Unsupported memory object type: {type(memory_object).__name__}"
        )


class WriteBackService:
    def __init__(
        self,
        repository_router: MemoryRepositoryRouter,
        resolver: Optional[WritebackPolicyResolver] = None,
        write_policies: Optional[list[MemoryWritePolicy]] = None,
    ):
        self._resolver = resolver if resolver else WritebackPolicyResolver(
                [
                    FactWritebackPolicy(),
                    EpisodeWritebackPolicy(),
                    PreferenceWritebackPolicy(),
                    NoteWritebackPolicy(),# always in the end, because she is fallback
                ]
            )
        self._write_policies = write_policies if write_policies else  [
            ProvenancePolicy(),
            TTLPolicy(),
            PiiPolicy(),
            ConflictPolicy(reject_on_conflict=True),
            ConfidencePolicy(),
            DedupPolicy(),
        ]
        self._repository_router = repository_router 

    def write(self, request: WritebackRequest) -> WritebackResult:
        result = WritebackResult()

        context = WritebackContext(
            source=request.source,
            dry_run=request.dry_run,
            meta=request.meta,
            vector_repo=self._repository_router.vector_repo,
            graph_repo=self._repository_router.graph_repo,
            episodic_repo=self._repository_router.episodic_repo,
        )

        for item in request.items:
            try:
                conversion_policy = self._resolver.resolve(item)
                memory_object = conversion_policy.convert(item)

                for policy in self._write_policies:
                    decision = policy.apply(memory_object, context)

                    if decision.rejected:
                        result.add_rejected(decision)
                        break

                    if decision.memory_object is None:
                        result.add_error(
                            WritebackDecision.reject(
                                reason="policy_returned_empty_memory_object",
                                policy=policy.name,
                            )
                        )
                        break

                    memory_object = decision.memory_object
                else:
                    if not request.dry_run:
                        self._repository_router.save(memory_object)

                    result.add_saved(memory_object)

            except Exception as exc:
                result.add_error(
                    WritebackDecision.reject(
                        reason="writeback_error",
                        detail=str(exc),
                        policy="WriteBackService",
                    )
                )

        return result
    
    # for legacy support
    def write_raw(self, raw_items: list[dict[str, Any]]) -> WritebackResult:
        request = WritebackRequest.model_validate({"items": raw_items})
        return self.write(request)