from __future__ import annotations

import time
from typing import Any

from sqlalchemy import desc, select

from app.config import settings
from domain.writeback import WritebackResult
from infra.db.models import (
    MemoryOperationRow,
    MemoryRecordRow,
    PolicyDecisionRow,
    ReviewCandidateRow,
)
from infra.db.session import DatabaseHandle


class SqlAuditRepository:
    def __init__(self, database: DatabaseHandle) -> None:
        self.database = database

    def save_writeback_result(self, result: WritebackResult) -> None:
        now = time.time()
        with self.database.session() as db:
            for memory_object in result.saved:
                data = memory_object.model_dump(mode="json")
                provenance = data.get("provenance") or {}
                meta = data.get("meta") or {}
                row = MemoryRecordRow(
                    memory_id=str(data.get("id")),
                    memory_kind=_memory_kind(data),
                    status="saved",
                    subject=data.get("subject"),
                    predicate=data.get("predicate"),
                    object=data.get("object"),
                    source=provenance.get("source"),
                    object_json=data,
                    meta_json=meta,
                    created_at=float(provenance.get("time") or now),
                    updated_at=now,
                )
                db.merge(row)

            for operation in result.operations:
                data = operation.model_dump(mode="json")
                db.merge(
                    MemoryOperationRow(
                        id=data["id"],
                        operation=data["operation"],
                        status=data["status"],
                        source=data.get("source"),
                        item_id=data.get("item_id"),
                        memory_id=data.get("memory_id"),
                        memory_kind=data.get("memory_kind"),
                        before_json=data.get("before"),
                        after_json=data.get("after"),
                        meta_json=data.get("meta") or {},
                        created_at=float(data.get("created_at") or now),
                        completed_at=data.get("completed_at"),
                    )
                )

            for decision in result.policy_decisions:
                data = decision.model_dump(mode="json")
                db.merge(
                    PolicyDecisionRow(
                        id=data["id"],
                        operation_id=data.get("operation_id"),
                        stage=data["stage"],
                        policy=data["policy"],
                        action=data["action"],
                        accepted=bool(data["accepted"]),
                        item_id=data.get("item_id"),
                        memory_id=data.get("memory_id"),
                        memory_kind=data.get("memory_kind"),
                        reason=data.get("reason"),
                        detail=data.get("detail"),
                        meta_json=data.get("meta") or {},
                        created_at=float(data.get("created_at") or now),
                    )
                )

            for rejected in result.rejected:
                if rejected.reason != "requires_review":
                    continue
                memory_object = rejected.memory_object
                object_json = memory_object.model_dump(mode="json") if memory_object is not None else None
                candidate_id = f"review_{rejected.id or int(now * 1000)}"
                operation_id = _operation_id_for_memory(result, rejected.id)
                db.merge(
                    ReviewCandidateRow(
                        id=candidate_id,
                        operation_id=operation_id,
                        status="pending_review",
                        item_id=_item_id_for_operation(result, operation_id),
                        memory_id=rejected.id,
                        memory_kind=rejected.kind,
                        source=_source_for_operation(result, operation_id),
                        reason=rejected.reason,
                        detail=rejected.detail,
                        memory_object_json=object_json,
                        meta_json=rejected.meta or {},
                        created_at=now,
                    )
                )

    def list_operations(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        memory_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.database.session() as db:
            stmt = select(MemoryOperationRow).order_by(desc(MemoryOperationRow.created_at)).limit(limit)
            if status:
                stmt = stmt.where(MemoryOperationRow.status == status)
            if memory_id:
                stmt = stmt.where(MemoryOperationRow.memory_id == memory_id)
            return [_row_to_dict(row) for row in db.scalars(stmt).all()]

    def list_policy_decisions(
        self,
        *,
        limit: int = 50,
        operation_id: str | None = None,
        policy: str | None = None,
        accepted: bool | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.database.session() as db:
            stmt = select(PolicyDecisionRow).order_by(desc(PolicyDecisionRow.created_at)).limit(limit)
            if operation_id:
                stmt = stmt.where(PolicyDecisionRow.operation_id == operation_id)
            if policy:
                stmt = stmt.where(PolicyDecisionRow.policy == policy)
            if accepted is not None:
                stmt = stmt.where(PolicyDecisionRow.accepted == accepted)
            return [_row_to_dict(row) for row in db.scalars(stmt).all()]

    def list_review_candidates(
        self,
        *,
        limit: int = 50,
        status: str | None = "pending_review",
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.database.session() as db:
            stmt = select(ReviewCandidateRow).order_by(desc(ReviewCandidateRow.created_at)).limit(limit)
            if status:
                stmt = stmt.where(ReviewCandidateRow.status == status)
            return [_row_to_dict(row) for row in db.scalars(stmt).all()]

    def list_memory_records(
        self,
        *,
        limit: int = 50,
        memory_kind: str | None = None,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.database.session() as db:
            stmt = select(MemoryRecordRow).order_by(desc(MemoryRecordRow.updated_at)).limit(limit)
            if memory_kind:
                stmt = stmt.where(MemoryRecordRow.memory_kind == memory_kind)
            if subject:
                stmt = stmt.where(MemoryRecordRow.subject == subject)
            return [_row_to_dict(row) for row in db.scalars(stmt).all()]


def build_audit_repository() -> SqlAuditRepository | None:
    if not settings.memory_audit_enabled or not settings.memory_database_url:
        return None
    handle = DatabaseHandle(settings.memory_database_url)
    if settings.memory_audit_auto_create:
        handle.create_all()
    return SqlAuditRepository(handle)


def _memory_kind(data: dict[str, Any]) -> str | None:
    if "subject" in data and "predicate" in data and "object" in data:
        return "fact"
    return data.get("type")


def _operation_id_for_memory(result: WritebackResult, memory_id: str | None) -> str | None:
    for operation in result.operations:
        if operation.memory_id == memory_id:
            return operation.id
    return result.operations[0].id if result.operations else None


def _item_id_for_operation(result: WritebackResult, operation_id: str | None) -> str | None:
    for operation in result.operations:
        if operation.id == operation_id:
            return operation.item_id
    return None


def _source_for_operation(result: WritebackResult, operation_id: str | None) -> str | None:
    for operation in result.operations:
        if operation.id == operation_id:
            return operation.source
    return None


def _row_to_dict(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in row.__table__.columns:
        out[column.name] = getattr(row, column.name)
    return out
