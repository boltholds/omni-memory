
from typing import Any, Protocol
from domain.models import (
    DecisionRecord,
    Episode,
    EpisodeEvent,
    ExperienceRecord,
    Fact,
    FailurePatternRecord,
    MemoryObject,
    Provenance,
    SkillRecord,
)
from domain.writeback import (
    DomainMemoryObject,
    WritebackConversionPolicy,
    WritebackRawItem,
    clean_meta,
    get_item_id,
    normalize_provenance,
)
from domain.policy import MemoryPolicy


class WritebackPolicyNotFoundError(ValueError):
    pass


class WritebackPolicyResolver:
    def __init__(self, policies: list[WritebackConversionPolicy]) -> None:
        if not policies:
            raise ValueError("At least one writeback conversion policy is required")

        self._policies = policies

    def resolve(self, item: WritebackRawItem) -> WritebackConversionPolicy:
        for policy in self._policies:
            if policy.matches(item):
                return policy

        raise WritebackPolicyNotFoundError("No writeback conversion policy matched item")


def _parse(self, raw: dict[str, Any]) -> tuple[str, DomainMemoryObject]:
    item = WritebackRawItem.model_validate(raw)
    policy = self._resolver.resolve(item)
    return policy.kind, policy.convert(item, memory_policy=self._policy)


class FactWritebackPolicy:
    name = "fact_writeback"
    kind = "fact"

    def matches(self, item: WritebackRawItem) -> bool:
        if item.type == "fact":
            return True

        payload = item.payload or {}

        return (
            item.subject is not None
            and item.predicate is not None
            and item.object is not None
        ) or (
            payload.get("subject") is not None
            and payload.get("predicate") is not None
            and payload.get("object") is not None
        )

    def convert(self, item: WritebackRawItem) -> Fact:
        payload = item.payload or {}

        return Fact(
            id=get_item_id(item, prefix="fact"),
            subject=item.subject or str(payload.get("subject", "")),
            predicate=item.predicate or str(payload.get("predicate", "")),
            object=item.object or str(payload.get("object", "")),
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class EpisodeWritebackPolicy:
    name = "episode_writeback"
    kind = "episode"

    def matches(self, item: WritebackRawItem) -> bool:
        if item.type == "episode":
            return True

        payload = item.payload or {}

        return (
            item.participants is not None
            or item.events is not None
            or payload.get("participants") is not None
            or payload.get("events") is not None
        )

    def convert(self, item: WritebackRawItem) -> Episode:
        payload = item.payload or {}

        raw_events = item.events
        if raw_events is None:
            raw_events = payload.get("events", [])

        events = [
            event if isinstance(event, EpisodeEvent) else EpisodeEvent.model_validate(event)
            for event in raw_events or []
        ]

        return Episode(
            id=get_item_id(item, prefix="episode"),
            participants=item.participants or payload.get("participants", []),
            summary=item.summary or payload.get("summary", ""),
            events=events,
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class DecisionWritebackPolicy:
    name = "decision_writeback"
    kind = "decision"

    def matches(self, item: WritebackRawItem) -> bool:
        return item.type in {"decision", "adr"}

    def convert(self, item: WritebackRawItem) -> DecisionRecord:
        payload = item.payload or {}

        return DecisionRecord(
            id=get_item_id(item, prefix="decision"),
            title=item.summary or str(payload.get("title") or payload.get("summary") or ""),
            status=str(payload.get("status") or item.meta.get("status") or "accepted"),
            context=str(payload.get("context") or item.content or ""),
            decision=str(payload.get("decision") or item.text or ""),
            consequences=list(payload.get("consequences") or []),
            alternatives=list(payload.get("alternatives") or []),
            refs=dict(payload.get("refs") or {}),
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class ExperienceWritebackPolicy:
    name = "experience_writeback"
    kind = "experience"

    def matches(self, item: WritebackRawItem) -> bool:
        return item.type == "experience"

    def convert(self, item: WritebackRawItem) -> ExperienceRecord:
        payload = item.payload or {}

        return ExperienceRecord(
            id=get_item_id(item, prefix="experience"),
            goal=str(payload.get("goal") or item.summary or ""),
            context=str(payload.get("context") or item.content or ""),
            decision=str(payload.get("decision") or ""),
            actions=list(payload.get("actions") or []),
            outcome=str(payload.get("outcome") or ""),
            evaluation=dict(payload.get("evaluation") or {}),
            lesson=str(payload.get("lesson") or item.text or ""),
            reuse_when=list(payload.get("reuse_when") or []),
            avoid_when=list(payload.get("avoid_when") or []),
            confidence=float(payload.get("confidence") or item.meta.get("confidence") or 0.5),
            refs=dict(payload.get("refs") or {}),
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class SkillWritebackPolicy:
    name = "skill_writeback"
    kind = "skill"

    def matches(self, item: WritebackRawItem) -> bool:
        return item.type == "skill"

    def convert(self, item: WritebackRawItem) -> SkillRecord:
        payload = item.payload or {}
        return SkillRecord(
            id=get_item_id(item, prefix="skill"),
            name=str(payload.get("name") or item.summary or ""),
            problem=str(payload.get("problem") or item.content or ""),
            procedure=list(payload.get("procedure") or []),
            reuse_when=list(payload.get("reuse_when") or []),
            avoid_when=list(payload.get("avoid_when") or []),
            evidence_ids=list(payload.get("evidence_ids") or []),
            confidence=float(payload.get("confidence") or item.meta.get("confidence") or 0.5),
            refs=dict(payload.get("refs") or {}),
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class FailurePatternWritebackPolicy:
    name = "failure_pattern_writeback"
    kind = "failure_pattern"

    def matches(self, item: WritebackRawItem) -> bool:
        return item.type in {"failure_pattern", "failure-pattern"}

    def convert(self, item: WritebackRawItem) -> FailurePatternRecord:
        payload = item.payload or {}
        return FailurePatternRecord(
            id=get_item_id(item, prefix="failure_pattern"),
            symptom=str(payload.get("symptom") or item.summary or ""),
            root_cause=str(payload.get("root_cause") or ""),
            fix=str(payload.get("fix") or item.text or ""),
            detection=str(payload.get("detection") or ""),
            evidence_ids=list(payload.get("evidence_ids") or []),
            confidence=float(payload.get("confidence") or item.meta.get("confidence") or 0.5),
            refs=dict(payload.get("refs") or {}),
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class NoteWritebackPolicy:
    name = "note_writeback"
    kind = "note"

    def matches(self, item: WritebackRawItem) -> bool:
        return True

    def convert(self, item: WritebackRawItem) -> MemoryObject:
        payload = dict(item.payload or {})

        if "text" not in payload and item.text is not None:
            payload["text"] = item.text

        if "text" not in payload and item.content is not None:
            payload["text"] = item.content

        if not payload:
            payload = {"raw": item.model_dump(exclude_none=True)}

        return MemoryObject(
            id=get_item_id(item, prefix="note"),
            type=item.type or "note",
            payload=payload,
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )


class PreferenceWritebackPolicy:
    name = "preference_writeback"
    kind = "preference"

    def matches(self, item: WritebackRawItem) -> bool:
        if item.type == "preference":
            return True

        payload = item.payload or {}

        return (
            payload.get("type") == "preference"
            or payload.get("kind") == "preference"
            or "preference" in payload
        )

    def convert(self, item: WritebackRawItem) -> MemoryObject:
        payload = dict(item.payload or {})

        if "text" not in payload and item.text is not None:
            payload["text"] = item.text

        if "text" not in payload and item.content is not None:
            payload["text"] = item.content

        payload.setdefault("kind", "preference")

        return MemoryObject(
            id=get_item_id(item, prefix="pref"),
            type="preference",
            payload=payload,
            provenance=normalize_provenance(item),
            meta=clean_meta(item),
        )
