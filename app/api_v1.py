from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from infra.db.audit_repo import build_audit_repository
from infra.metrics import metrics


PolicyMode = Literal["permissive", "strict", "review"]


class MemoryRememberIn(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "api"
    dry_run: bool = False
    policy_mode: PolicyMode = "permissive"
    meta: dict[str, Any] = Field(default_factory=dict)


class MemorySearchIn(BaseModel):
    q: str
    k_sem: int = 5
    k_eps: int = 3
    intent: str | None = None
    mode: str | None = None


class MemoryContextIn(BaseModel):
    q: str = ""
    max_tokens: int | None = None
    lang: Literal["en", "ru"] = "en"
    style: Literal["concise", "bullets", "detailed", "plain"] = "concise"
    intent: str | None = None
    mode: str | None = None


def build_v1_router(memory, orchestrator) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1-memory"])
    audit_repo = build_audit_repository()

    def audit_status() -> dict[str, Any]:
        return {
            "enabled": audit_repo is not None,
            "configured": bool(settings.memory_database_url),
            "auto_create": settings.memory_audit_auto_create,
        }

    def assemble_context_for_query(
        q: str,
        max_tokens: int | None = None,
        intent: str | None = None,
        mode: str | None = None,
    ):
        bundle = orchestrator.plan_retrieval(q, intent=intent, mode=mode)
        if max_tokens:
            old = settings.context_max_tokens
            settings.context_max_tokens = int(max_tokens)
            try:
                pack = orchestrator.assemble_context(bundle, intent=intent, mode=mode)
            finally:
                settings.context_max_tokens = old
        else:
            pack = orchestrator.assemble_context(bundle, intent=intent, mode=mode)
        return bundle, pack

    @router.post("/memories/remember")
    def remember(inp: MemoryRememberIn):
        """Write memory items and return auditable writeback details."""
        meta = dict(inp.meta or {})
        meta["policy_mode"] = inp.policy_mode

        result = memory.write_items_raw(
            inp.items,
            source=inp.source,
            dry_run=inp.dry_run,
            meta=meta,
        )

        persisted = False
        persistence_error = None
        if audit_repo is not None:
            try:
                audit_repo.save_writeback_result(result)
                persisted = True
            except Exception as exc:
                persistence_error = f"{type(exc).__name__}: {exc}"

        metrics.inc("v1_memory_remember_calls", 1)
        metrics.inc("writeback_saved", result.saved_count)
        metrics.inc("writeback_rejected", result.rejected_count)

        payload = result.model_dump(mode="json")
        payload["audit_persistence"] = {
            **audit_status(),
            "persisted": persisted,
            "error": persistence_error,
        }
        return payload

    @router.post("/memories/search")
    def search(inp: MemorySearchIn):
        """Retrieve semantic chunks, graph facts, current beliefs and episodes."""
        bundle = memory.retrieve(
            inp.q,
            k_sem=inp.k_sem,
            k_eps=inp.k_eps,
            intent=inp.intent,
            mode=inp.mode,
        )
        metrics.inc("v1_memory_search_calls", 1)
        return bundle.model_dump(mode="json")

    @router.post("/context")
    def context(inp: MemoryContextIn):
        """Return structured context plus the retrieval bundle used to build it."""
        bundle, pack = assemble_context_for_query(
            inp.q,
            inp.max_tokens,
            intent=inp.intent,
            mode=inp.mode,
        )
        metrics.inc("v1_context_calls", 1)
        return {
            "query": inp.q,
            "context": pack.model_dump(mode="json"),
            "retrieval": bundle.model_dump(mode="json"),
        }

    return router
