from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.agent_cycle import AgentCycleRecord, experience_from_agent_cycle
from domain.models import WriteReport
from domain.writeback import WritebackRawItem, WritebackRequest, WritebackResult


@dataclass(frozen=True)
class WriteItemsCommand:
    """Compatibility wrapper for building a WritebackRequest.

    This is no longer part of a command interpreter layer. Public facade methods
    call WriteBackService directly and use this class only as a small request
    factory for raw item batches.
    """

    items: list[dict[str, Any]]
    source: str = "user"
    dry_run: bool = False
    meta: dict[str, Any] | None = None

    def to_request(self) -> WritebackRequest:
        return WritebackRequest(
            source=self.source,
            dry_run=self.dry_run,
            meta=self.meta or {},
            items=[WritebackRawItem.model_validate(item) for item in self.items],
        )


class SingleWritebackItemCommand:
    """Base class for one-memory-item factories.

    The historical `Command` suffix is kept for compatibility with existing
    imports. The class only builds raw writeback items; it does not execute
    writeback on its own.
    """

    item_type: str
    id_prefix: str
    source: str
    meta: dict[str, Any] | None

    def payload(self) -> dict[str, Any]:
        return {}

    def top_level_fields(self) -> dict[str, Any]:
        return {}

    def item_meta(self) -> dict[str, Any]:
        return dict(self.meta or {})

    def to_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": f"{self.id_prefix}-{uuid.uuid4().hex}",
            "type": self.item_type,
            "provenance": {"source": self.source, "time": time.time(), "meta": {}},
            "meta": self.item_meta(),
        }
        payload = self.payload()
        if payload:
            item["payload"] = payload
        item.update(self.top_level_fields())
        return item


@dataclass(frozen=True)
class WriteFactCommand(SingleWritebackItemCommand):
    subject: str
    predicate: str
    object_: str
    source: str = "user"
    confidence: float = 1.0
    meta: dict[str, Any] | None = None

    item_type = "fact"
    id_prefix = "fact"

    def top_level_fields(self) -> dict[str, Any]:
        return {
            "subject": self.subject.lower().strip(),
            "predicate": self.predicate.lower().strip(),
            "object": self.object_.lower().strip(),
        }

    def item_meta(self) -> dict[str, Any]:
        meta = dict(self.meta or {})
        meta.setdefault("confidence", self.confidence)
        return meta


@dataclass(frozen=True)
class WriteNoteCommand(SingleWritebackItemCommand):
    text: str
    source: str = "user"
    meta: dict[str, Any] | None = None

    item_type = "note"
    id_prefix = "note"

    def payload(self) -> dict[str, Any]:
        return {"text": self.text}


@dataclass(frozen=True)
class WriteDecisionCommand(SingleWritebackItemCommand):
    title: str
    decision: str
    context: str = ""
    consequences: list[str] | None = None
    alternatives: list[str] | None = None
    refs: dict[str, Any] | None = None
    status: str = "accepted"
    source: str = "user"
    meta: dict[str, Any] | None = None

    item_type = "decision"
    id_prefix = "decision"

    def payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences or [],
            "alternatives": self.alternatives or [],
            "refs": self.refs or {},
        }


@dataclass(frozen=True)
class RecordExperienceCommand(SingleWritebackItemCommand):
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

    item_type = "experience"
    id_prefix = "experience"

    def payload(self) -> dict[str, Any]:
        return {
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
        }


@dataclass(frozen=True)
class WriteSkillCommand(SingleWritebackItemCommand):
    name: str
    problem: str = ""
    procedure: list[str] | None = None
    reuse_when: list[str] | None = None
    avoid_when: list[str] | None = None
    evidence_ids: list[str] | None = None
    confidence: float = 0.5
    refs: dict[str, Any] | None = None
    source: str = "user"
    meta: dict[str, Any] | None = None

    item_type = "skill"
    id_prefix = "skill"

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "problem": self.problem,
            "procedure": self.procedure or [],
            "reuse_when": self.reuse_when or [],
            "avoid_when": self.avoid_when or [],
            "evidence_ids": self.evidence_ids or [],
            "confidence": self.confidence,
            "refs": self.refs or {},
        }


@dataclass(frozen=True)
class WriteFailurePatternCommand(SingleWritebackItemCommand):
    symptom: str
    root_cause: str = ""
    fix: str = ""
    detection: str = ""
    evidence_ids: list[str] | None = None
    confidence: float = 0.5
    refs: dict[str, Any] | None = None
    source: str = "user"
    meta: dict[str, Any] | None = None

    item_type = "failure_pattern"
    id_prefix = "failure-pattern"

    def payload(self) -> dict[str, Any]:
        return {
            "symptom": self.symptom,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "detection": self.detection,
            "evidence_ids": self.evidence_ids or [],
            "confidence": self.confidence,
            "refs": self.refs or {},
        }


@dataclass(frozen=True)
class RecordAgentCycleCommand:
    cycle: AgentCycleRecord | dict[str, Any]
    source: str = "agent-cycle"

    def to_item(self) -> dict[str, Any]:
        record = self.cycle if isinstance(self.cycle, AgentCycleRecord) else AgentCycleRecord.model_validate(self.cycle)
        payload = experience_from_agent_cycle(record)
        meta = dict(payload.pop("meta") or {})
        meta.setdefault("recorded_from", "agent_cycle")
        return RecordExperienceCommand(**payload, source=self.source, meta=meta).to_item()


def to_write_report(result: WritebackResult) -> WriteReport:
    return WriteReport(
        saved=result.saved_count,
        rejected=result.rejected_count + result.error_count,
        reasons=result.reasons,
    )
