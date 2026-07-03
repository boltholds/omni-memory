from __future__ import annotations

import time
import re


from typing import Any

from pydantic import BaseModel
from domain.models import Provenance, Fact, Episode, MemoryObject

from app.memory_hygiene import normalize_memory_scope_meta
from domain.writeback import (
    MemoryWritePolicy,
    WritebackContext,
    WritebackDecision,
    extract_text,
    text_signature,
)

DomainMemoryObject = MemoryObject | Fact | Episode

class ProvenancePolicy(MemoryWritePolicy):
    """
    Гарантирует, что у объекта есть provenance and canonical memory scope.

    Если source/time не заполнены, подставляет batch source и текущее время.
    Scope хранится в meta["scope"], чтобы не ломать старые Pydantic-схемы.
    """

    name = "provenance"

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        provenance = memory_object.provenance or Provenance()

        source = provenance.source or context.source or "user"
        timestamp = provenance.time or time.time()

        provenance_meta = dict(provenance.meta or {})
        if context.meta:
            provenance_meta.setdefault("writeback", {})
            provenance_meta["writeback"].update(context.meta)

        object_meta = normalize_memory_scope_meta(
            dict(memory_object.meta or {}),
            source=source,
            context_meta=dict(context.meta or {}),
            memory_object=memory_object,
        )

        updated = memory_object.model_copy(
            update={
                "provenance": Provenance(
                    source=source,
                    time=timestamp,
                    meta=provenance_meta,
                ),
                "meta": object_meta,
            }
        )

        return WritebackDecision.accept(updated, policy=self.name)


class TTLConfig(BaseModel):
    high_volatility_days: int = 7
    normal_days: int = 365


class TTLPolicy(MemoryWritePolicy):
    """
    Добавляет meta.expire_at, если его ещё нет.

    Правило:
    - meta.volatility == "high" -> короткий TTL;
    - иначе normal TTL.
    """

    name = "ttl"

    def __init__(self, config: TTLConfig | None = None) -> None:
        self.config = config or TTLConfig()

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        meta = dict(memory_object.meta or {})

        if "expire_at" in meta:
            return WritebackDecision.accept(memory_object, policy=self.name)

        volatility = str(meta.get("volatility", "normal")).lower()

        days = (
            self.config.high_volatility_days
            if volatility == "high"
            else self.config.normal_days
        )

        meta["expire_at"] = time.time() + days * 86400.0

        updated = memory_object.model_copy(update={"meta": meta})

        return WritebackDecision.accept(
            updated,
            policy=self.name,
            meta={
                "ttl_days": days,
                "volatility": volatility,
            },
        )


class PiiPolicy(MemoryWritePolicy):
    """
    Блокирует запись объектов, в тексте которых похожие email/API keys/secrets.

    Сейчас политика простая и локальная.
    Потом её можно заменить на более строгий PII detector.
    """

    name = "pii"

    email_re = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )
    secret_re = re.compile(
        r"(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
        re.IGNORECASE,
    )

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        text = extract_text(memory_object)[:5000]

        if self.email_re.search(text):
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="pii_email_blocked",
                policy=self.name,
            )

        if self.secret_re.search(text):
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="pii_secret_blocked",
                policy=self.name,
            )

        return WritebackDecision.accept(memory_object, policy=self.name)


class ConfidenceConfig(BaseModel):
    accept: float = 0.6
    reject: float = 0.3
    default_fact_confidence: float = 0.5
    reject_when_missing: bool = False


class ConfidencePolicy(MemoryWritePolicy):
    """
    Проверяет confidence для фактов.

    Источники confidence:
    1. memory_object.meta["confidence"]
    2. memory_object.meta["score"]
    3. default_fact_confidence

    По умолчанию отсутствующий confidence НЕ блокирует факт жёстко,
    чтобы не сломать старое поведение.
    """

    name = "confidence"

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self.config = config or ConfidenceConfig()

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        if not isinstance(memory_object, Fact):
            return WritebackDecision.accept(memory_object, policy=self.name)

        meta = dict(memory_object.meta or {})

        raw_confidence = meta.get("confidence", meta.get("score"))

        if raw_confidence is None:
            if self.config.reject_when_missing:
                return WritebackDecision.reject(
                    memory_object=memory_object,
                    reason="missing_confidence",
                    policy=self.name,
                )

            confidence = self.config.default_fact_confidence
        else:
            confidence = float(raw_confidence)

        if confidence < self.config.accept:
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="low_confidence",
                detail=f"confidence={confidence:.2f}; threshold={self.config.accept:.2f}",
                policy=self.name,
                meta={
                    "confidence": confidence,
                    "threshold": self.config.accept,
                },
            )

        return WritebackDecision.accept(
            memory_object,
            policy=self.name,
            meta={"confidence": confidence},
        )


class DedupPolicy(MemoryWritePolicy):
    """
    Дедуплицирует текстовые MemoryObject внутри batch и против vector_repo.
    Also rejects exact duplicate Fact objects already present in graph_repo.

    Ожидает, что vector_repo может иметь метод:
        is_duplicate_text(text: str) -> bool

    Если метода нет, проверяется только текущий batch.
    """

    name = "dedup"

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        if isinstance(memory_object, Fact):
            graph_repo = context.graph_repo
            if graph_repo is not None and hasattr(graph_repo, "query"):
                existing_facts = graph_repo.query(
                    subject=memory_object.subject,
                    predicate=memory_object.predicate,
                )
                duplicates = [
                    fact
                    for fact in existing_facts
                    if fact.object == memory_object.object
                ]
                if duplicates:
                    return WritebackDecision.reject(
                        memory_object=memory_object,
                        reason="duplicate_fact",
                        policy=self.name,
                        meta={
                            "duplicates": [
                                {
                                    "id": fact.id,
                                    "subject": fact.subject,
                                    "predicate": fact.predicate,
                                    "object": fact.object,
                                }
                                for fact in duplicates
                            ]
                        },
                    )
            return WritebackDecision.accept(memory_object, policy=self.name)

        if not isinstance(memory_object, MemoryObject):
            return WritebackDecision.accept(memory_object, policy=self.name)

        text = extract_text(memory_object)

        if not text:
            return WritebackDecision.accept(memory_object, policy=self.name)

        signature = text_signature(text)

        if signature in context.seen_note_signatures:
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="duplicate_in_batch",
                policy=self.name,
                meta={"signature": signature},
            )

        vector_repo = context.vector_repo

        if (
            vector_repo is not None
            and hasattr(vector_repo, "is_duplicate_text")
            and vector_repo.is_duplicate_text(text)
        ):
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="duplicate_in_repository",
                policy=self.name,
                meta={"signature": signature},
            )

        context.seen_note_signatures.add(signature)

        return WritebackDecision.accept(
            memory_object,
            policy=self.name,
            meta={"signature": signature},
        )


class ConflictPolicy(MemoryWritePolicy):
    """
    Проверяет конфликт фактов вида:

        same subject + same predicate + different object

    Policy modes can be controlled via WritebackRequest.meta["policy_mode"]:
    - permissive: save the incoming fact and attach conflict metadata;
    - strict: reject the incoming conflicting fact;
    - review: reject with requires_review so a future UI can approve/edit it.
    """

    name = "conflict"

    def __init__(self, *, reject_on_conflict: bool = True) -> None:
        self.reject_on_conflict = reject_on_conflict

    def _mode(self, context: WritebackContext) -> str:
        mode = str((context.meta or {}).get("policy_mode", "")).lower().strip()
        if mode in {"permissive", "strict", "review"}:
            return mode
        return "strict" if self.reject_on_conflict else "permissive"

    def apply(
        self,
        memory_object: DomainMemoryObject,
        context: WritebackContext,
    ) -> WritebackDecision:
        if not isinstance(memory_object, Fact):
            return WritebackDecision.accept(memory_object, policy=self.name)

        graph_repo = context.graph_repo

        if graph_repo is None or not hasattr(graph_repo, "query"):
            return WritebackDecision.accept(memory_object, policy=self.name)

        existing_facts = graph_repo.query(
            subject=memory_object.subject,
            predicate=memory_object.predicate,
        )

        conflicts = [
            fact
            for fact in existing_facts
            if fact.object != memory_object.object
        ]

        if not conflicts:
            return WritebackDecision.accept(memory_object, policy=self.name)

        conflict_meta = {
            "mode": self._mode(context),
            "incoming": {
                "id": memory_object.id,
                "subject": memory_object.subject,
                "predicate": memory_object.predicate,
                "object": memory_object.object,
            },
            "conflicts": [
                {
                    "id": fact.id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                }
                for fact in conflicts
            ],
        }

        mode = self._mode(context)
        if mode == "strict":
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="fact_conflict",
                policy=self.name,
                meta=conflict_meta,
            )

        if mode == "review":
            return WritebackDecision.reject(
                memory_object=memory_object,
                reason="requires_review",
                detail="conflicting_fact_requires_human_review",
                policy=self.name,
                meta=conflict_meta,
            )

        meta = dict(memory_object.meta or {})
        meta["conflict"] = conflict_meta

        updated = memory_object.model_copy(update={"meta": meta})

        return WritebackDecision.accept(
            updated,
            policy=self.name,
            meta=conflict_meta,
        )
