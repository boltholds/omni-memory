from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from omni_memory.config import settings
from omni_memory.infra.db.audit_repo import build_audit_repository
from omni_memory.infra.metrics import metrics


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
    scope: dict[str, Any] = Field(default_factory=dict)


class MemoryContextIn(BaseModel):
    q: str = ""
    max_tokens: int | None = None
    lang: Literal["en", "ru"] = "en"
    style: Literal["concise", "bullets", "detailed", "plain"] = "concise"
    intent: str | None = None
    mode: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict)


class FactMineIn(BaseModel):
    text: str
    source: str = "api-fact-mining"
    dry_run: bool = True
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    policy_mode: PolicyMode = "review"
    domain_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


def build_v1_router(memory, orchestrator) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1-memory"])
    audit_repo = build_audit_repository()

    def audit_status() -> dict[str, Any]:
        return {
            "enabled": audit_repo is not None,
            "configured": bool(settings.memory_database_url),
            "auto_create": settings.memory_audit_auto_create,
        }

    def audit_items(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"audit_persistence": audit_status(), "items": items}

    def assemble_context_for_query(
        q: str,
        max_tokens: int | None = None,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ):
        bundle = orchestrator.plan_retrieval(q, intent=intent, mode=mode, scope=scope)
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

    @router.post("/facts/mine")
    def mine_facts(inp: FactMineIn):
        """Extract evidence-grounded fact candidates. Dry-run by default."""
        result = memory.mine_facts(
            inp.text,
            source=inp.source,
            dry_run=inp.dry_run,
            min_confidence=inp.min_confidence,
            policy_mode=inp.policy_mode,
            domain_ids=inp.domain_ids,
            meta=inp.meta,
        )
        persisted = False
        persistence_error = None
        if audit_repo is not None and result.writeback.operations:
            try:
                audit_repo.save_writeback_result(result.writeback)
                persisted = True
            except Exception as exc:
                persistence_error = f"{type(exc).__name__}: {exc}"
        payload = result.model_dump(mode="json")
        payload["audit_persistence"] = {
            **audit_status(),
            "persisted": persisted,
            "error": persistence_error,
        }
        metrics.inc("v1_fact_mining_calls", 1)
        metrics.inc("fact_mining_candidates", result.candidate_count)
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
            scope=inp.scope,
        )
        metrics.inc("v1_memory_search_calls", 1)
        return bundle.model_dump(mode="json")

    @router.get("/memories")
    def list_memory_records(limit: int = settings.memory_audit_default_limit):
        if audit_repo is None:
            return audit_items([])
        return audit_items(audit_repo.list_memory_records(limit=limit))

    @router.get("/audit/operations")
    def list_operations(limit: int = settings.memory_audit_default_limit):
        if audit_repo is None:
            return audit_items([])
        return audit_items(audit_repo.list_operations(limit=limit))

    @router.get("/audit/decisions")
    def list_decisions(limit: int = settings.memory_audit_default_limit):
        if audit_repo is None:
            return audit_items([])
        return audit_items(audit_repo.list_policy_decisions(limit=limit))

    @router.get("/audit/reviews")
    def list_reviews(limit: int = settings.memory_audit_default_limit):
        if audit_repo is None:
            return audit_items([])
        return audit_items(audit_repo.list_review_candidates(limit=limit))

    @router.post("/context")
    def context(inp: MemoryContextIn):
        """Return structured context plus the retrieval bundle used to build it."""
        bundle, pack = assemble_context_for_query(
            inp.q,
            inp.max_tokens,
            intent=inp.intent,
            mode=inp.mode,
            scope=inp.scope,
        )
        metrics.inc("v1_context_calls", 1)
        return {
            "query": inp.q,
            "context": pack.model_dump(mode="json"),
            "retrieval": bundle.model_dump(mode="json"),
        }

    return router
