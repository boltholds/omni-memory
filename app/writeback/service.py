from typing import Any, Optional, Protocol, runtime_checkable

from domain.models import Fact, Episode, MemoryObject
from domain.operations import MemoryOperation, PolicyDecision

from domain.writeback import (
    WritebackRawItem,
    WritebackResult,
    WritebackRequest,
    WritebackDecision,
    MemoryWritePolicy,
    DomainMemoryObject,
    WritebackContext,
    get_memory_kind,
)


from domain.ports import IMemoryWriteRepository
from app.writeback.writeback_policies import (
    FactWritebackPolicy,
    EpisodeWritebackPolicy,
    PreferenceWritebackPolicy,
    NoteWritebackPolicy,
    WritebackPolicyResolver,
)

from app.writeback.memory_policies import (
    ProvenancePolicy,
    TTLPolicy,
    PiiPolicy,
    ConflictPolicy,
    ConfidencePolicy,
    DedupPolicy,
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


def _raw_item_id(item: WritebackRawItem) -> str | None:
    return item.id or item.uuid or item.hash


def _dump_memory_object(memory_object: DomainMemoryObject | None) -> dict[str, Any] | None:
    if memory_object is None:
        return None
    try:
        return memory_object.model_dump(mode="json")
    except Exception:
        return {"repr": repr(memory_object)}


def _policy_decision_from_writeback(
    decision: WritebackDecision,
    *,
    operation: MemoryOperation,
    stage: str = "write_policy",
    fallback_policy: str = "unknown",
) -> PolicyDecision:
    action = "accept" if decision.accepted else "reject"
    return PolicyDecision(
        operation_id=operation.id,
        stage=stage,  # type: ignore[arg-type]
        policy=decision.policy or fallback_policy,
        action=action,  # type: ignore[arg-type]
        accepted=decision.accepted,
        item_id=operation.item_id,
        memory_id=decision.id or operation.memory_id,
        memory_kind=decision.kind or operation.memory_kind,
        reason=decision.reason,
        detail=decision.detail,
        meta=decision.meta,
    )


def _service_policy_decision(
    *,
    operation: MemoryOperation,
    policy: str,
    action: str,
    accepted: bool,
    reason: str | None = None,
    detail: str | None = None,
    memory_object: DomainMemoryObject | None = None,
    meta: dict[str, Any] | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        operation_id=operation.id,
        stage="service",
        policy=policy,
        action=action,  # type: ignore[arg-type]
        accepted=accepted,
        item_id=operation.item_id,
        memory_id=getattr(memory_object, "id", operation.memory_id),
        memory_kind=get_memory_kind(memory_object) or operation.memory_kind,
        reason=reason,
        detail=detail,
        meta=meta or {},
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
                    NoteWritebackPolicy(),  # always in the end, because she is fallback
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
            operation = MemoryOperation(
                operation="remember",
                status="started",
                source=request.source,
                item_id=_raw_item_id(item),
                before=item.model_dump(mode="json", exclude_none=True),
                meta={"dry_run": request.dry_run, **dict(request.meta or {})},
            )

            try:
                conversion_policy = self._resolver.resolve(item)
                memory_object = conversion_policy.convert(item)
                operation = operation.model_copy(
                    update={
                        "memory_id": getattr(memory_object, "id", None),
                        "memory_kind": get_memory_kind(memory_object),
                    }
                )
                result.add_policy_decision(
                    PolicyDecision(
                        operation_id=operation.id,
                        stage="conversion",
                        policy=conversion_policy.name,
                        action="accept",
                        accepted=True,
                        item_id=operation.item_id,
                        memory_id=getattr(memory_object, "id", None),
                        memory_kind=get_memory_kind(memory_object),
                        meta={"kind": conversion_policy.kind},
                    )
                )

                for policy in self._write_policies:
                    decision = policy.apply(memory_object, context)
                    result.add_policy_decision(
                        _policy_decision_from_writeback(
                            decision,
                            operation=operation,
                            fallback_policy=policy.name,
                        )
                    )

                    if decision.rejected:
                        result.add_rejected(decision)
                        operation = operation.complete(
                            "rejected",
                            memory_id=decision.id,
                            memory_kind=decision.kind,
                            after=_dump_memory_object(decision.memory_object),
                            meta={
                                "rejected_by": decision.policy or policy.name,
                                "reason": decision.reason,
                            },
                        )
                        break

                    if decision.memory_object is None:
                        error_decision = WritebackDecision.reject(
                            reason="policy_returned_empty_memory_object",
                            policy=policy.name,
                        )
                        result.add_error(error_decision)
                        result.add_policy_decision(
                            _policy_decision_from_writeback(
                                error_decision,
                                operation=operation,
                                fallback_policy=policy.name,
                            )
                        )
                        operation = operation.complete(
                            "error",
                            meta={
                                "error_policy": policy.name,
                                "reason": "policy_returned_empty_memory_object",
                            },
                        )
                        break

                    memory_object = decision.memory_object
                    operation = operation.model_copy(
                        update={
                            "memory_id": getattr(memory_object, "id", None),
                            "memory_kind": get_memory_kind(memory_object),
                        }
                    )
                else:
                    if request.dry_run:
                        result.add_policy_decision(
                            _service_policy_decision(
                                operation=operation,
                                policy="repository",
                                action="skip",
                                accepted=True,
                                reason="dry_run",
                                memory_object=memory_object,
                            )
                        )
                        operation = operation.complete(
                            "accepted",
                            memory_id=getattr(memory_object, "id", None),
                            memory_kind=get_memory_kind(memory_object),
                            after=_dump_memory_object(memory_object),
                            meta={"dry_run": True},
                        )
                    else:
                        self._repository_router.save(memory_object)
                        result.add_policy_decision(
                            PolicyDecision(
                                operation_id=operation.id,
                                stage="repository",
                                policy="repository_router",
                                action="save",
                                accepted=True,
                                item_id=operation.item_id,
                                memory_id=getattr(memory_object, "id", None),
                                memory_kind=get_memory_kind(memory_object),
                            )
                        )
                        operation = operation.complete(
                            "saved",
                            memory_id=getattr(memory_object, "id", None),
                            memory_kind=get_memory_kind(memory_object),
                            after=_dump_memory_object(memory_object),
                        )

                    result.add_saved(memory_object)

            except Exception as exc:
                error_decision = WritebackDecision.reject(
                    reason="writeback_error",
                    detail=str(exc),
                    policy="WriteBackService",
                )
                result.add_error(error_decision)
                result.add_policy_decision(
                    _service_policy_decision(
                        operation=operation,
                        policy="WriteBackService",
                        action="error",
                        accepted=False,
                        reason="writeback_error",
                        detail=str(exc),
                    )
                )
                operation = operation.complete(
                    "error",
                    meta={"reason": "writeback_error", "detail": str(exc)},
                )
            finally:
                result.add_operation(operation)

        return result

    # for legacy support
    def write_raw(self, raw_items: list[dict[str, Any]]) -> WritebackResult:
        request = WritebackRequest.model_validate({"items": raw_items})
        return self.write(request)
