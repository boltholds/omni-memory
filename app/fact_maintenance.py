from __future__ import annotations

import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

from domain.models import Fact, Provenance
from domain.repositories import IFactRepo


class FactMaintenanceCommand(BaseModel):
    operation: str
    fact_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    status: str | None = None
    patch: dict[str, Any] = Field(default_factory=dict)
    new_fact: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    source: str = "maintenance"
    dry_run: bool = False
    limit: int | None = None


class FactMaintenanceResult(BaseModel):
    operation: str
    applied: bool = False
    fact: Fact | None = None
    facts: list[Fact] = Field(default_factory=list)
    removed: int = 0
    reasons: list[str] = Field(default_factory=list)
    dry_run: bool = False


class FactMaintenanceContext(BaseModel):
    repo: Any

    model_config = {"arbitrary_types_allowed": True}


class FactMaintenanceStrategy(Protocol):
    name: str

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        ...


class ListFactsStrategy:
    name = "list"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        facts = context.repo.query(
            subject=command.subject,
            predicate=command.predicate,
            object=command.object,
        )
        if command.status:
            facts = [fact for fact in facts if _fact_status(fact) == command.status]
        if command.limit is not None:
            facts = facts[: max(0, command.limit)]
        return FactMaintenanceResult(
            operation=self.name,
            applied=True,
            facts=facts,
            dry_run=command.dry_run,
        )


class GetFactStrategy:
    name = "get"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        if not command.fact_id:
            return _missing_id(self.name, command)
        fact = context.repo.get_fact(command.fact_id)
        return FactMaintenanceResult(
            operation=self.name,
            applied=fact is not None,
            fact=fact,
            reasons=[] if fact else [f"Fact not found: {command.fact_id}"],
            dry_run=command.dry_run,
        )


class PatchFactStrategy:
    name = "patch"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        if not command.fact_id:
            return _missing_id(self.name, command)
        fact = context.repo.get_fact(command.fact_id)
        if fact is None:
            return _not_found(self.name, command)

        updated = _patch_fact(fact, command.patch)
        if command.reason:
            updated.meta["reason"] = command.reason
        if not command.dry_run:
            context.repo.save_fact(updated)
        return FactMaintenanceResult(
            operation=self.name,
            applied=not command.dry_run,
            fact=updated,
            dry_run=command.dry_run,
        )


class RetractFactStrategy:
    name = "retract"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        if not command.fact_id:
            return _missing_id(self.name, command)
        fact = context.repo.get_fact(command.fact_id)
        if fact is None:
            return _not_found(self.name, command)

        updated = fact.model_copy(deep=True)
        updated.meta["status"] = "retracted"
        updated.meta["retracted_at"] = time.time()
        if command.reason:
            updated.meta["reason"] = command.reason
        if not command.dry_run:
            context.repo.save_fact(updated)
        return FactMaintenanceResult(
            operation=self.name,
            applied=not command.dry_run,
            fact=updated,
            dry_run=command.dry_run,
        )


class SupersedeFactStrategy:
    name = "supersede"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        if not command.fact_id:
            return _missing_id(self.name, command)
        old = context.repo.get_fact(command.fact_id)
        if old is None:
            return _not_found(self.name, command)

        new = _build_superseding_fact(old, command)
        historical = old.model_copy(deep=True)
        historical.meta["status"] = "historical"
        historical.meta["superseded_by"] = new.id
        historical.meta["valid_to"] = historical.meta.get("valid_to", time.time())
        if command.reason:
            historical.meta["reason"] = command.reason

        if not command.dry_run:
            context.repo.save_fact(historical)
            context.repo.save_fact(new)

        return FactMaintenanceResult(
            operation=self.name,
            applied=not command.dry_run,
            fact=new,
            facts=[historical, new],
            dry_run=command.dry_run,
        )


class HardDeleteFactStrategy:
    name = "hard_delete"

    def apply(
        self,
        command: FactMaintenanceCommand,
        context: FactMaintenanceContext,
    ) -> FactMaintenanceResult:
        if not command.fact_id:
            return _missing_id(self.name, command)
        fact = context.repo.get_fact(command.fact_id)
        if fact is None:
            return _not_found(self.name, command)
        removed = 0 if command.dry_run else int(context.repo.remove_fact(command.fact_id))
        return FactMaintenanceResult(
            operation=self.name,
            applied=removed > 0,
            fact=fact,
            removed=removed,
            dry_run=command.dry_run,
        )


class FactMaintenanceService:
    def __init__(
        self,
        repo: IFactRepo,
        strategies: list[FactMaintenanceStrategy] | None = None,
    ) -> None:
        self._context = FactMaintenanceContext(repo=repo)
        selected = strategies or [
            ListFactsStrategy(),
            GetFactStrategy(),
            PatchFactStrategy(),
            RetractFactStrategy(),
            SupersedeFactStrategy(),
            HardDeleteFactStrategy(),
        ]
        self._strategies = {strategy.name: strategy for strategy in selected}

    def execute(self, command: FactMaintenanceCommand | dict[str, Any]) -> FactMaintenanceResult:
        parsed = (
            command
            if isinstance(command, FactMaintenanceCommand)
            else FactMaintenanceCommand.model_validate(command)
        )
        strategy = self._strategies.get(parsed.operation)
        if strategy is None:
            return FactMaintenanceResult(
                operation=parsed.operation,
                applied=False,
                reasons=[f"Unsupported fact maintenance operation: {parsed.operation}"],
                dry_run=parsed.dry_run,
            )
        return strategy.apply(parsed, self._context)


def _missing_id(operation: str, command: FactMaintenanceCommand) -> FactMaintenanceResult:
    return FactMaintenanceResult(
        operation=operation,
        applied=False,
        reasons=["fact_id is required"],
        dry_run=command.dry_run,
    )


def _not_found(operation: str, command: FactMaintenanceCommand) -> FactMaintenanceResult:
    return FactMaintenanceResult(
        operation=operation,
        applied=False,
        reasons=[f"Fact not found: {command.fact_id}"],
        dry_run=command.dry_run,
    )


def _fact_status(fact: Fact) -> str:
    return str((fact.meta or {}).get("status") or "current")


def _patch_fact(fact: Fact, patch: dict[str, Any]) -> Fact:
    data = fact.model_dump(mode="python")
    for key in ("subject", "predicate", "object"):
        if key in patch:
            data[key] = str(patch[key])
    if "meta" in patch:
        data["meta"] = {**data.get("meta", {}), **(patch.get("meta") or {})}
    if "provenance" in patch:
        data["provenance"] = {**data.get("provenance", {}), **(patch.get("provenance") or {})}
    return Fact.model_validate(data)


def _build_superseding_fact(old: Fact, command: FactMaintenanceCommand) -> Fact:
    raw = command.new_fact or {}
    now = time.time()
    meta = {
        **(raw.get("meta") or {}),
        "status": "current",
        "supersedes": [old.id],
    }
    if command.reason:
        meta["reason"] = command.reason

    provenance = raw.get("provenance") or {
        "source": command.source,
        "time": now,
        "meta": {},
    }

    return Fact(
        id=str(raw.get("id") or f"fact-{uuid.uuid4().hex}"),
        subject=str(raw.get("subject") or command.subject or old.subject),
        predicate=str(raw.get("predicate") or command.predicate or old.predicate),
        object=str(raw.get("object") or command.object or old.object),
        provenance=Provenance.model_validate(provenance),
        meta=meta,
    )
