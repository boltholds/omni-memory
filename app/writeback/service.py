from typing import Any, Optional, Protocol, runtime_checkable

from domain.models import (
    DecisionRecord,
    Episode,
    ExperienceRecord,
    Fact,
    FailurePatternRecord,
    MemoryObject,
    SkillRecord,
)
from domain.operations import MemoryOperation, PolicyDecision

from domain.writeback import (
    WritebackRawItem,
    WritebackResult,
    WritebackRequest,
    WritebackDecision,
    MemoryWritePolicy,
    DomainMemoryObject,
    WritebackContext,
    get_item_id,
    get_memory_kind,
)

from domain.ports import IMemoryWriteRepository
from app.writeback.writeback_policies import (
    DecisionWritebackPolicy,
    EpisodeWritebackPolicy,
    ExperienceWritebackPolicy,
    FactWritebackPolicy,
    FailurePatternWritebackPolicy,
    NoteWritebackPolicy,
    PreferenceWritebackPolicy,
    SkillWritebackPolicy,
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
from app.telemetry import span as telemetry_span


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


@runtime_checkable
class DecisionMemoryRepository(Protocol):
    def save_decision(self, decision: DecisionRecord) -> None:
        ...


@runtime_checkable
class ExperienceMemoryRepository(Protocol):
    def save_experience(self, experience: ExperienceRecord) -> None:
        ...


@runtime_checkable
class SkillMemoryRepository(Protocol):
    def save_skill(self, skill: SkillRecord) -> None:
        ...


@runtime_checkable
class FailurePatternMemoryRepository(Protocol):
    def save_failure_pattern(self, pattern: FailurePatternRecord) -> None:
        ...


class MemoryRepositoryRouter:
    def __init__(
        self,
        *,
        vector_repo: VectorMemoryRepository,
        graph_repo: GraphMemoryRepository,
        episodic_repo: EpisodicMemoryRepository,
        decision_repo: DecisionMemoryRepository | None = None,
        experience_repo: ExperienceMemoryRepository | None = None,
        skill_repo: SkillMemoryRepository | None = None,
        failure_pattern_repo: FailurePatternMemoryRepository | None = None,
    ) -> None:
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo
        self.episodic_repo = episodic_repo
        self.decision_repo = decision_repo
        self.experience_repo = experience_repo
        self.skill_repo = skill_repo
        self.failure_pattern_repo = failure_pattern_repo

    def save(self, memory_object: DomainMemoryObject) -> None:
        if isinstance(memory_object, Fact):
            self.graph_repo.save_fact(memory_object)
            return

        if isinstance(memory_object, Episode):
            self.episodic_repo.save_episode(memory_object)
            return

        if isinstance(memory_object, DecisionRecord):
            if self.decision_repo is None:
                raise TypeError("Decision repository is not configured")
            self.decision_repo.save_decision(memory_object)
            return

        if isinstance(memory_object, ExperienceRecord):
            if self.experience_repo is None:
                raise TypeError("Experience repository is not configured")
            self.experience_repo.save_experience(memory_object)
            return

        if isinstance(memory_object, SkillRecord):
            if self.skill_repo is None:
                raise TypeError("Skill repository is not configured")
            self.skill_repo.save_skill(memory_object)
            return

        if isinstance(memory_object, FailurePatternRecord):
            if self.failure_pattern_repo is None:
                raise TypeError("Failure pattern repository is not configured")
            self.failure_pattern_repo.save_failure_pattern(memory_object)
            return

        if isinstance(memory_object, MemoryObject):
            self.vector_repo.save_object(memory_object)
            return

        raise TypeError(f"Unsupported memory object type: {type(memory_object).__name__}")


def _raw_item_id(item: WritebackRawItem) -> str:
    return get_item_id(item, prefix="item")


def _dump_memory_object(memory_object: DomainMemoryObject | None) -> dict[str, Any] | None:
    if memory_object is None:
        return None
    try:
        return memory_object.model_dump(mode="json")
    except Exception:
        return {"repr": repr(memory_object)}


def _apply_request_source_if_implicit(
    memory_object: DomainMemoryObject,
    *,
    item: WritebackRawItem,
    request_source: str | None,
) -> DomainMemoryObject:
    if item.provenance is not None or not request_source:
        return memory_object

    provenance = getattr(memory_object, "provenance", None)
    if provenance is None:
        return memory_object

    updated_provenance = provenance.model_copy(update={"source": request_source})
    return memory_object.model_copy(update={"provenance": updated_provenance})


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
                    DecisionWritebackPolicy(),
                    ExperienceWritebackPolicy(),
                    SkillWritebackPolicy(),
                    FailurePatternWritebackPolicy(),
                    NoteWritebackPolicy(),
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
        with telemetry_span("writeback.write", item_count=len(request.items), source=request.source, dry_run=request.dry_run) as span:
            result = self._write(request)
            _set_span_attribute(span, "saved_count", result.saved_count)
            _set_span_attribute(span, "rejected_count", result.rejected_count)
            _set_span_attribute(span, "error_count", result.error_count)
            return result

    def _write(self, request: WritebackRequest) -> WritebackResult:
        result = WritebackResult()

        context = WritebackContext(
            source=request.source,
            dry_run=request.dry_run,
            meta=request.meta,
            vector_repo=self._repository_router.vector_repo,
            graph_repo=self._repository_router.graph_repo,
            episodic_repo=self._repository_router.episodic_repo,
            decision_repo=self._repository_router.decision_repo,
            experience_repo=self._repository_router.experience_repo,
            skill_repo=self._repository_router.skill_repo,
            failure_pattern_repo=self._repository_router.failure_pattern_repo,
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
                memory_object = _apply_request_source_if_implicit(
                    memory_object,
                    item=item,
                    request_source=request.source,
                )
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

    def write_raw(self, raw_items: list[dict[str, Any]]) -> WritebackResult:
        request = WritebackRequest.model_validate({"items": raw_items})
        return self.write(request)


def _set_span_attribute(span: Any | None, key: str, value: Any) -> None:
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute(key, value)
