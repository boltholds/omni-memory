from __future__ import annotations

import re
from typing import Any

from omni_memory.domain.models import MemoryScope
from omni_memory.domain.writeback import DomainMemoryObject, MemoryWritePolicy, WritebackContext, WritebackDecision

_TEST_SOURCE_MARKERS = {"test", "tests", "pytest", "unit-test", "unit-tests"}
_BENCHMARK_SOURCE_MARKERS = {"benchmark", "bench", "memory-eval", "eval"}
_ALICE_FIXTURE_RE = re.compile(r"^alice[-_][0-9a-f]{8,}", re.IGNORECASE)


class MemoryHygienePolicy(MemoryWritePolicy):
    """Attach canonical scope metadata and quarantine obvious test artifacts.

    This is intentionally a write-path policy, not a storage primitive. It keeps
    the current data model backward-compatible by storing scope under
    `memory_object.meta["scope"]` instead of adding hard required fields to every
    memory record class.
    """

    name = "memory_hygiene"

    def apply(self, memory_object: DomainMemoryObject, context: WritebackContext) -> WritebackDecision:
        meta = normalize_memory_scope_meta(
            dict(getattr(memory_object, "meta", {}) or {}),
            source=_source(memory_object, context),
            context_meta=dict(context.meta or {}),
            memory_object=memory_object,
        )
        updated = memory_object.model_copy(update={"meta": meta})
        return WritebackDecision.accept(
            updated,
            policy=self.name,
            meta={"scope": meta.get("scope")},
        )


def normalize_memory_scope_meta(
    meta: dict[str, Any],
    *,
    source: str | None,
    context_meta: dict[str, Any] | None = None,
    memory_object: Any | None = None,
) -> dict[str, Any]:
    context_meta = context_meta or {}
    scope_raw = _as_dict(context_meta.get("scope")) | _as_dict(meta.get("scope"))

    source_value = str(source or "").strip().lower()
    category = str(meta.get("category") or context_meta.get("category") or "").strip().lower()

    inferred_environment = _first_str(
        scope_raw.get("environment"),
        meta.get("environment"),
        context_meta.get("environment"),
    )
    inferred_durability = _first_str(
        scope_raw.get("durability"),
        meta.get("durability"),
        context_meta.get("durability"),
    )

    is_test = _is_test_source(source_value, category) or _looks_like_test_fixture(memory_object)
    is_benchmark = _is_benchmark_source(source_value, category)

    if is_test:
        inferred_environment = "test"
        inferred_durability = "ephemeral"
        meta["volatility"] = "high"
    elif is_benchmark:
        inferred_environment = "benchmark"
        inferred_durability = "ephemeral"
        meta["volatility"] = "high"

    domain_ids = _string_list(
        scope_raw.get("domain_ids")
        or meta.get("domain_ids")
        or meta.get("domains")
        or context_meta.get("domain_ids")
        or context_meta.get("domains")
    )

    exclude_from_consolidation = bool(
        scope_raw.get("exclude_from_consolidation")
        or meta.get("exclude_from_consolidation")
        or context_meta.get("exclude_from_consolidation")
        or is_test
        or is_benchmark
        or inferred_durability in {"ephemeral", "session"}
    )

    scope = MemoryScope(
        tenant_id=str(scope_raw.get("tenant_id") or meta.get("tenant_id") or context_meta.get("tenant_id") or "default"),
        agent_id=_optional_str(scope_raw.get("agent_id") or meta.get("agent_id") or context_meta.get("agent_id")),
        domain_ids=domain_ids,
        environment=_normalize_environment(inferred_environment),
        durability=_normalize_durability(inferred_durability),
        visibility=_normalize_visibility(scope_raw.get("visibility") or meta.get("visibility") or context_meta.get("visibility")),
        exclude_from_consolidation=exclude_from_consolidation,
    )

    warnings = list(meta.get("scope_warnings") or [])
    if (
        scope.durability == "durable"
        and scope.visibility != "global"
        and not scope.domain_ids
    ):
        warnings.append("durable_memory_without_domain")

    meta["scope"] = scope.model_dump(mode="json")
    meta["exclude_from_consolidation"] = scope.exclude_from_consolidation
    if warnings:
        meta["scope_warnings"] = _unique(warnings)
    return meta


def _source(memory_object: Any, context: WritebackContext) -> str | None:
    provenance = getattr(memory_object, "provenance", None)
    if provenance is not None and getattr(provenance, "source", None):
        return provenance.source
    return context.source


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _first_str(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    return _unique([str(item).strip() for item in values if str(item).strip()])


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _is_test_source(source: str, category: str) -> bool:
    return (
        source in _TEST_SOURCE_MARKERS
        or source.startswith("test-")
        or source.endswith("-test")
        or category == "test"
    )


def _is_benchmark_source(source: str, category: str) -> bool:
    return (
        source in _BENCHMARK_SOURCE_MARKERS
        or "benchmark" in source
        or category in {"benchmark", "eval", "memory-eval"}
    )


def _looks_like_test_fixture(memory_object: Any | None) -> bool:
    if memory_object is None:
        return False
    subject = str(getattr(memory_object, "subject", "") or "")
    predicate = str(getattr(memory_object, "predicate", "") or "")
    obj = str(getattr(memory_object, "object", "") or "")
    return bool(_ALICE_FIXTURE_RE.match(subject) and predicate == "at" and obj == "lighthouse")


def _normalize_environment(value: str | None) -> str:
    normalized = str(value or "dev").strip().lower().replace("-", "_")
    aliases = {"ci": "test", "tests": "test", "pytest": "test", "bench": "benchmark"}
    return aliases.get(normalized, normalized) if normalized in {"prod", "dev", "test", "benchmark", "sandbox"} or normalized in aliases else "dev"


def _normalize_durability(value: str | None) -> str:
    normalized = str(value or "durable").strip().lower().replace("-", "_")
    aliases = {"temporary": "ephemeral", "transient": "ephemeral", "runtime": "ephemeral"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"durable", "ephemeral", "session"} else "durable"


def _normalize_visibility(value: str | None) -> str:
    normalized = str(value or "private").strip().lower().replace("-", "_")
    return normalized if normalized in {"private", "shared", "global"} else "private"
