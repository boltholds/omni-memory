from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class PolicyDecision(BaseModel):
    """A single auditable decision made while processing memory.

    This is intentionally storage-agnostic. Today it can be returned in API
    responses; later the same model can be persisted to an audit log table.
    """

    id: str = Field(default_factory=lambda: _new_id("poldec"))
    operation_id: str | None = None

    stage: Literal["conversion", "write_policy", "repository", "service"] = "write_policy"
    policy: str
    action: Literal["accept", "reject", "error", "save", "skip"]
    accepted: bool

    item_id: str | None = None
    memory_id: str | None = None
    memory_kind: str | None = None

    reason: str | None = None
    detail: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class MemoryCandidate(BaseModel):
    """A proposed memory before or during writeback.

    Candidates are useful for future human-in-the-loop review flows: the system
    can propose a memory, attach evidence and policy decisions, and wait for
    approval before writing it to long-term storage.
    """

    id: str = Field(default_factory=lambda: _new_id("cand"))
    operation_id: str | None = None
    status: Literal["proposed", "accepted", "rejected", "error"] = "proposed"

    item_id: str | None = None
    memory_id: str | None = None
    memory_kind: str | None = None
    memory_object: dict[str, Any] | None = None

    source: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    reason: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class MemoryOperation(BaseModel):
    """Audit envelope for a memory operation.

    Examples: remember, recall, update, forget, expire, merge, context.
    """

    id: str = Field(default_factory=lambda: _new_id("op"))
    operation: Literal[
        "remember",
        "recall",
        "update",
        "forget",
        "expire",
        "merge",
        "context",
        "search",
    ]
    status: Literal["started", "accepted", "rejected", "saved", "error", "skipped"] = "started"

    source: str | None = None
    actor_id: str | None = None
    request_id: str | None = None

    item_id: str | None = None
    memory_id: str | None = None
    memory_kind: str | None = None

    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None

    def complete(
        self,
        status: Literal["accepted", "rejected", "saved", "error", "skipped"],
        *,
        memory_id: str | None = None,
        memory_kind: str | None = None,
        after: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "MemoryOperation":
        merged_meta = dict(self.meta or {})
        if meta:
            merged_meta.update(meta)

        return self.model_copy(
            update={
                "status": status,
                "memory_id": memory_id or self.memory_id,
                "memory_kind": memory_kind or self.memory_kind,
                "after": after if after is not None else self.after,
                "meta": merged_meta,
                "completed_at": time.time(),
            }
        )
