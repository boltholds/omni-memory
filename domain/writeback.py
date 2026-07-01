
import hashlib
import json
import re

from typing import Any, Dict, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field, ConfigDict
from domain.models import Fact, Episode ,Provenance, MemoryObject
from domain.operations import MemoryOperation, PolicyDecision


DomainMemoryObject = MemoryObject | Fact | Episode


class WritebackContext(BaseModel):
    """
    Контекст одной batch writeback-операции.

    Нужен политикам, которым требуется состояние:
    - dedup внутри текущего batch;
    - доступ к репозиторию;
    - batch-level meta;
    - dry_run;
    - source.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str | None = None
    dry_run: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    vector_repo: Any | None = None
    graph_repo: Any | None = None
    episodic_repo: Any | None = None

    seen_note_signatures: set[str] = Field(default_factory=set)


class WritebackRawItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    uuid: str | None = None
    hash: str | None = None

    type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    text: str | None = None
    content: str | None = None

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None

    participants: list[str] | None = None
    summary: str | None = None
    events: list[dict[str, Any]] | None = None

    provenance:  Provenance | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    


class WritebackSavedItem(BaseModel):
    kind: Literal["note", "fact", "episode"]
    id: str


class WritebackRejectedItem(BaseModel):
    kind: str | None = None
    id: str | None = None
    reason: str
    detail: str | None = None



class WritebackDecision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    accepted: bool
    memory_object: DomainMemoryObject | None = None

    kind: str | None = None
    id: str | None = None

    reason: str | None = None
    detail: str | None = None
    policy: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.accepted

    @classmethod
    def accept(
        cls,
        memory_object: DomainMemoryObject,
        *,
        policy: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "WritebackDecision":
        return cls(
            accepted=True,
            memory_object=memory_object,
            kind=get_memory_kind(memory_object),
            id=getattr(memory_object, "id", None),
            policy=policy,
            meta=meta or {},
        )

    @classmethod
    def reject(
        cls,
        *,
        reason: str,
        memory_object: DomainMemoryObject | None = None,
        kind: str | None = None,
        id: str | None = None,
        detail: str | None = None,
        policy: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "WritebackDecision":
        return cls(
            accepted=False,
            memory_object=memory_object,
            kind=kind or get_memory_kind(memory_object),
            id=id or getattr(memory_object, "id", None),
            reason=reason,
            detail=detail,
            policy=policy,
            meta=meta or {},
        )


class WritebackResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    saved: list[DomainMemoryObject] = Field(default_factory=list)
    rejected: list[WritebackDecision] = Field(default_factory=list)
    errors: list[WritebackDecision] = Field(default_factory=list)

    policy_decisions: list[PolicyDecision] = Field(default_factory=list)
    operations: list[MemoryOperation] = Field(default_factory=list)

    @property
    def saved_count(self) -> int:
        return len(self.saved)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return self.saved_count + self.rejected_count + self.error_count

    @property
    def reasons(self) -> list[str]:
        """
        Совместимость со старым WriteReport.reasons.
        """
        result: list[str] = []

        for decision in [*self.rejected, *self.errors]:
            reason = decision.reason or "unknown"
            if decision.id:
                result.append(f"{reason}:{decision.id}")
            else:
                result.append(reason)

        return result

    def add_saved(self, memory_object: DomainMemoryObject) -> None:
        self.saved.append(memory_object)

    def add_rejected(self, decision: WritebackDecision) -> None:
        self.rejected.append(decision)

    def add_error(self, decision: WritebackDecision) -> None:
        self.errors.append(decision)

    def add_policy_decision(self, decision: PolicyDecision) -> None:
        self.policy_decisions.append(decision)

    def add_operation(self, operation: MemoryOperation) -> None:
        self.operations.append(operation)


class WritebackRequest(BaseModel):
    items: list[WritebackRawItem] = Field(default_factory=list)

    source: str | None = None
    dry_run: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)





@runtime_checkable
class WritebackConversionPolicy(Protocol):
    name: str
    kind: str

    def matches(self, item: WritebackRawItem) -> bool:
        ...

    def convert(self, item: WritebackRawItem) -> MemoryObject:
        ...



@runtime_checkable
class MemoryWritePolicy(Protocol):
    name: str

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        ...

# Временно для совместимости со старым кодом:
WriteReport = WritebackResult


def normalize_provenance(item: WritebackRawItem) -> Provenance:
    if item.provenance is None:
        return Provenance()

    if isinstance(item.provenance, Provenance):
        return item.provenance

    return Provenance.model_validate(item.provenance)


def clean_meta(item: WritebackRawItem) -> dict[str, Any]:
    return dict(item.meta or {})


def stable_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def get_item_id(item: WritebackRawItem, *, prefix: str) -> str:
    return item.id or item.uuid or item.hash or stable_id(
        prefix,
        item.model_dump(exclude_none=True),
    )
    
    
def get_memory_kind(memory_object: DomainMemoryObject | None) -> str | None:
    if memory_object is None:
        return None

    if isinstance(memory_object, Fact):
        return "fact"

    if isinstance(memory_object, Episode):
        return "episode"

    if isinstance(memory_object, MemoryObject):
        return memory_object.type

    return type(memory_object).__name__


def extract_text(memory_object: DomainMemoryObject) -> str:
    if isinstance(memory_object, MemoryObject):
        payload = memory_object.payload or {}

        parts: list[str] = []

        for key in ("text", "content", "summary"):
            value = payload.get(key)
            if value:
                parts.append(str(value))

        return "\n".join(parts)

    if isinstance(memory_object, Fact):
        return f"{memory_object.subject} {memory_object.predicate} {memory_object.object}"

    if isinstance(memory_object, Episode):
        parts = [memory_object.summary]

        for event in memory_object.events:
            if event.summary:
                parts.append(event.summary)

        return "\n".join(part for part in parts if part)

    return ""


_norm_ws_re = re.compile(r"\s+")


def text_signature(text: str) -> str:
    normalized = _norm_ws_re.sub(" ", text.strip().lower())[:4096]
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
