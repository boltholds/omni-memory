from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from domain.models import Provenance, ReviewItem
from domain.writeback import stable_id

ReviewAction = Literal["accept", "reject", "supersede"]


@dataclass(frozen=True)
class ReviewActionResult:
    item: ReviewItem | None
    applied: bool = False
    result: dict[str, Any] | None = None
    created: ReviewItem | None = None
    reason: str = ""

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "item": self.item.model_dump(mode=mode) if self.item else None,
            "applied": self.applied,
            "result": self.result,
            "created": self.created.model_dump(mode=mode) if self.created else None,
            "reason": self.reason,
        }


class ReviewQueueService:
    def __init__(self, *, repo: Any, memory: Any) -> None:
        self.repo = repo
        self.memory = memory

    def submit(
        self,
        *,
        kind: str,
        title: str,
        payload: dict[str, Any],
        confidence: float = 0.5,
        reason: str = "",
        source: str = "review-queue",
        meta: dict[str, Any] | None = None,
    ) -> ReviewItem:
        normalized_kind = _normalize_kind(kind)
        item = ReviewItem(
            id=stable_id("review", {"kind": normalized_kind, "title": title, "payload": payload}),
            kind=normalized_kind,
            title=title,
            payload=dict(payload or {}),
            confidence=confidence,
            reason=reason,
            provenance=Provenance(source=source, time=time.time(), meta={}),
            meta=meta or {},
        )
        self.repo.save_review_item(item)
        return item

    def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewItem]:
        return self.repo.list_review_items(status=status, kind=kind, limit=limit)

    def get(self, item_id: str) -> ReviewItem | None:
        return self.repo.get_review_item(item_id)

    def accept(self, item_id: str, *, reviewer: str = "user", note: str = "") -> ReviewActionResult:
        item = self.repo.get_review_item(item_id)
        if item is None:
            return ReviewActionResult(item=None, reason="review_item_not_found")
        if item.status != "proposed":
            return ReviewActionResult(item=item, reason=f"review_item_not_proposed:{item.status}")

        result = self._apply(item)
        updated = _reviewed(item, status="accepted", reviewer=reviewer, note=note)
        self.repo.save_review_item(updated)
        return ReviewActionResult(item=updated, applied=True, result=result)

    def reject(self, item_id: str, *, reviewer: str = "user", note: str = "") -> ReviewActionResult:
        item = self.repo.get_review_item(item_id)
        if item is None:
            return ReviewActionResult(item=None, reason="review_item_not_found")
        if item.status != "proposed":
            return ReviewActionResult(item=item, reason=f"review_item_not_proposed:{item.status}")
        updated = _reviewed(item, status="rejected", reviewer=reviewer, note=note)
        self.repo.save_review_item(updated)
        return ReviewActionResult(item=updated, applied=True)

    def supersede(
        self,
        item_id: str,
        *,
        replacement: dict[str, Any],
        reviewer: str = "user",
        note: str = "",
    ) -> ReviewActionResult:
        item = self.repo.get_review_item(item_id)
        if item is None:
            return ReviewActionResult(item=None, reason="review_item_not_found")
        if item.status != "proposed":
            return ReviewActionResult(item=item, reason=f"review_item_not_proposed:{item.status}")

        replacement_item = self.submit(
            kind=replacement.get("kind") or item.kind,
            title=replacement.get("title") or item.title,
            payload=replacement.get("payload") or item.payload,
            confidence=float(replacement.get("confidence", item.confidence)),
            reason=replacement.get("reason") or note or item.reason,
            source=replacement.get("source", "review-queue"),
            meta={**item.meta, **dict(replacement.get("meta") or {})},
        )
        updated = _reviewed(
            item,
            status="superseded",
            reviewer=reviewer,
            note=note,
            superseded_by=replacement_item.id,
        )
        self.repo.save_review_item(updated)
        return ReviewActionResult(item=updated, applied=True, created=replacement_item)

    def _apply(self, item: ReviewItem) -> dict[str, Any]:
        payload = item.payload
        if item.kind == "decision":
            status = payload.get("status") or "accepted"
            return self.memory.write_decision(
                title=payload.get("title") or item.title,
                decision=payload.get("decision", ""),
                context=payload.get("context", ""),
                consequences=payload.get("consequences") or [],
                alternatives=payload.get("alternatives") or [],
                refs=payload.get("refs") or {},
                status=status if status != "proposed" else "accepted",
                source="review-queue",
                meta={**dict(payload.get("meta") or {}), "accepted_from_review_id": item.id},
            ).model_dump()
        if item.kind == "skill":
            return self.memory.write_skill(
                name=payload.get("name") or item.title,
                problem=payload.get("problem", ""),
                procedure=payload.get("procedure") or [],
                reuse_when=payload.get("reuse_when") or [],
                avoid_when=payload.get("avoid_when") or [],
                evidence_ids=payload.get("evidence_ids") or [],
                confidence=float(payload.get("confidence", item.confidence)),
                refs=payload.get("refs") or {},
                source="review-queue",
                meta={**dict(payload.get("meta") or {}), "accepted_from_review_id": item.id},
            ).model_dump()
        if item.kind == "failure_pattern":
            return self.memory.write_failure_pattern(
                symptom=payload.get("symptom") or item.title,
                root_cause=payload.get("root_cause", ""),
                fix=payload.get("fix", ""),
                detection=payload.get("detection", ""),
                evidence_ids=payload.get("evidence_ids") or [],
                confidence=float(payload.get("confidence", item.confidence)),
                refs=payload.get("refs") or {},
                source="review-queue",
                meta={**dict(payload.get("meta") or {}), "accepted_from_review_id": item.id},
            ).model_dump()
        if item.kind == "writeback_item":
            return self.memory.write_items(
                [payload],
                source="review-queue",
                dry_run=False,
                meta={"accepted_from_review_id": item.id},
            ).model_dump()
        raise ValueError(f"Unsupported review item kind: {item.kind}")


def _normalize_kind(kind: str) -> Literal["decision", "skill", "failure_pattern", "writeback_item"]:
    value = str(kind or "").strip().casefold().replace("-", "_")
    if value in {"decision", "adr", "decision_candidate"}:
        return "decision"
    if value in {"skill", "skill_candidate"}:
        return "skill"
    if value in {"failure_pattern", "failurepattern", "pattern", "failure_pattern_candidate"}:
        return "failure_pattern"
    if value in {"writeback_item", "memory_item", "fact", "note", "episode"}:
        return "writeback_item"
    raise ValueError(f"Unsupported review item kind: {kind}")


def _reviewed(
    item: ReviewItem,
    *,
    status: Literal["accepted", "rejected", "superseded"],
    reviewer: str,
    note: str = "",
    superseded_by: str | None = None,
) -> ReviewItem:
    meta = dict(item.meta or {})
    if note:
        meta["review_note"] = note
    return item.model_copy(
        update={
            "status": status,
            "reviewed_by": reviewer,
            "reviewed_at": time.time(),
            "superseded_by": superseded_by,
            "meta": meta,
        }
    )
