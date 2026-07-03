from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from app.agent_cycle import AgentCycleRecord, experience_from_agent_cycle
from app.writeback.service import WriteBackService
from domain.models import WriteReport
from domain.writeback import WritebackRawItem, WritebackRequest, WritebackResult


T = TypeVar("T")


@dataclass(frozen=True)
class MemoryCommandContext:
    writeback_service: WriteBackService


class MemoryCommand(Protocol[T]):
    def execute(self, context: MemoryCommandContext) -> T:
        ...


class MemoryCommandInterpreter:
    def __init__(self, context: MemoryCommandContext) -> None:
        self._context = context

    def execute(self, command: MemoryCommand[T]) -> T:
        return command.execute(self._context)


@dataclass(frozen=True)
class WriteItemsCommand:
    items: list[dict[str, Any]]
    source: str = "user"
    dry_run: bool = False
    meta: dict[str, Any] | None = None

    def execute(self, context: MemoryCommandContext) -> WritebackResult:
        request = WritebackRequest(
            source=self.source,
            dry_run=self.dry_run,
            meta=self.meta or {},
            items=[WritebackRawItem.model_validate(item) for item in self.items],
        )
        return context.writeback_service.write(request)


@dataclass(frozen=True)
class WriteFactCommand:
    subject: str
    predicate: str
    object_: str
    source: str = "user"
    confidence: float = 1.0

    def execute(self, context: MemoryCommandContext) -> WriteReport:
        item = {
            "id": f"fact-{uuid.uuid4().hex}",
            "type": "fact",
            "subject": self.subject.lower().strip(),
            "predicate": self.predicate.lower().strip(),
            "object": self.object_.lower().strip(),
            "provenance": {
                "source": self.source,
                "time": time.time(),
                "meta": {},
            },
            "meta": {
                "confidence": self.confidence,
            },
        }
        return _to_write_report(WriteItemsCommand([item], source=self.source).execute(context))


@dataclass(frozen=True)
class WriteNoteCommand:
    text: str
    source: str = "user"
    meta: dict[str, Any] | None = None

    def execute(self, context: MemoryCommandContext) -> WriteReport:
        item = {
            "id": f"note-{uuid.uuid4().hex}",
            "type": "note",
            "payload": {"text": self.text},
            "provenance": {
                "source": self.source,
                "time": time.time(),
                "meta": self.meta or {},
            },
            "meta": self.meta or {},
        }
        return _to_write_report(
            WriteItemsCommand([item], source=self.source, meta=self.meta).execute(context)
        )


@dataclass(frozen=True)
class WriteDecisionCommand:
    title: str
    decision: str
    context: str = ""
    consequences: list[str] | None = None
    alternatives: list[str] | None = None
    refs: dict[str, Any] | None = None
    status: str = "accepted"
    source: str = "user"
    meta: dict[str, Any] | None = None

    def execute(self, context: MemoryCommandContext) -> WriteReport:
        item = {
            "id": f"decision-{uuid.uuid4().hex}",
            "type": "decision",
            "payload": {
                "title": self.title,
                "status": self.status,
                "context": self.context,
                "decision": self.decision,
                "consequences": self.consequences or [],
                "alternatives": self.alternatives or [],
                "refs": self.refs or {},
            },
            "provenance": {
                "source": self.source,
                "time": time.time(),
                "meta": {},
            },
            "meta": self.meta or {},
        }
        return _to_write_report(
            WriteItemsCommand([item], source=self.source, meta=self.meta).execute(context)
        )


@dataclass(frozen=True)
class RecordExperienceCommand:
    goal: str
    lesson: str
    context: str = ""
    decision: str = ""
    actions: list[str] | None = None
    outcome: str = ""
    evaluation: dict[str, Any] | None = None
    reuse_when: list[str] | None = None
    avoid_when: list[str] | None = None
    confidence: float = 0.5
    refs: dict[str, Any] | None = None
    source: str = "user"
    meta: dict[str, Any] | None = None

    def execute(self, context: MemoryCommandContext) -> WriteReport:
        item = {
            "id": f"experience-{uuid.uuid4().hex}",
            "type": "experience",
            "payload": {
                "goal": self.goal,
                "context": self.context,
                "decision": self.decision,
                "actions": self.actions or [],
                "outcome": self.outcome,
                "evaluation": self.evaluation or {},
                "lesson": self.lesson,
                "reuse_when": self.reuse_when or [],
                "avoid_when": self.avoid_when or [],
                "confidence": self.confidence,
                "refs": self.refs or {},
            },
            "provenance": {
                "source": self.source,
                "time": time.time(),
                "meta": {},
            },
            "meta": self.meta or {},
        }
        return _to_write_report(
            WriteItemsCommand([item], source=self.source, meta=self.meta).execute(context)
        )


@dataclass(frozen=True)
class RecordAgentCycleCommand:
    cycle: AgentCycleRecord | dict[str, Any]
    source: str = "agent-cycle"

    def execute(self, context: MemoryCommandContext) -> WriteReport:
        record = (
            self.cycle
            if isinstance(self.cycle, AgentCycleRecord)
            else AgentCycleRecord.model_validate(self.cycle)
        )
        payload = experience_from_agent_cycle(record)
        meta = dict(payload.pop("meta") or {})
        meta.setdefault("recorded_from", "agent_cycle")
        return RecordExperienceCommand(
            **payload,
            source=self.source,
            meta=meta,
        ).execute(context)


def _to_write_report(result: WritebackResult) -> WriteReport:
    return WriteReport(
        saved=result.saved_count,
        rejected=result.rejected_count + result.error_count,
        reasons=result.reasons,
    )
