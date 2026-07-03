from __future__ import annotations

from sqlalchemy import Boolean, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from infra.db.base import Base


class MemoryRecordRow(Base):
    __tablename__ = "memory_records"

    memory_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    memory_kind: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="saved", index=True)
    subject: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    predicate: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    object: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    object_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    meta_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)


class MemoryOperationRow(Base):
    __tablename__ = "memory_operations"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_kind: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    meta_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    completed_at: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)


class PolicyDecisionRow(Base):
    __tablename__ = "policy_decisions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    policy: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_kind: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)


class ReviewCandidateRow(Base):
    __tablename__ = "review_candidates"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending_review", index=True)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_kind: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_object_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    meta_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
