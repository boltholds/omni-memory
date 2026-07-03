# infra/consistency.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple
import time
import unicodedata

from domain.models import ConflictItem, ConflictReport, Fact, FactBelief
from domain.ports import IConsistencyEngine


class SimpleConsistencyEngine(IConsistencyEngine):
    """
    Группируем факты по ключу (subject, predicate).
    Если у ключа больше 1 уникального object -> конфликт.
    """

    def detect_conflicts(self, facts: List[Fact]) -> ConflictReport:
        # key -> set(objects)
        groups: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        for f in facts:
            if _fact_status(f) in {"historical", "retracted"}:
                continue
            s = f.subject
            p = f.predicate
            o = f.object
            if not (s and p):
                # пропускаем некорректные факты; объект может быть пустым, но тогда не образует конфликт
                continue
            groups[(s, p)].add(o)

        conflicts: List[ConflictItem] = []
        for (s, p), objs in groups.items():
            variants = sorted([str(o) for o in objs if o is not None])
            if len(variants) > 1:
                conflicts.append(
                    ConflictItem(
                        key=f"{s}::{p}",
                        variants=variants,
                    )
                )
        # можно отсортировать для стабильности
        conflicts.sort(key=lambda c: c.key)
        return ConflictReport(conflicts=conflicts)


# --- опционально: простая эвристика доверия, пригодится позже ---

def score_trust_recent_first(facts: Iterable[Fact]) -> Dict[str, float]:
    """
    Примитивный скоринг доверия:
    - новее (provenance.time) -> выше
    - источник "user" < "system" < "verified"
    Возвращает map fact.id -> score ∈ [0, 1].
    """
    source_weight = {"user": 0.4, "system": 0.7, "verified": 1.0}
    items = list(facts)
    if not items:
        return {}

    times = [f.provenance.time or 0.0 for f in items]
    t_min, t_max = min(times), max(times)
    t_span = t_max - t_min

    scores: Dict[str, float] = {}
    # Если все времена одинаковые (в т.ч. ровно один факт) — считаем их максимально «свежими»
    uniform_time = t_span <= 1e-9

    for f in items:
        if uniform_time:
            t_norm = 1.0
        else:
            t_norm = ((f.provenance.time or 0.0) - t_min) / t_span  # 0..1

        s_norm = source_weight.get((f.provenance.source or "").lower(), 0.5)
        confidence = _float_meta(f.meta, "confidence", default=_float_meta(f.meta, "score", default=0.5))
        scores[f.id] = 0.4 * t_norm + 0.35 * s_norm + 0.25 * confidence
    return scores


def canonical_fact_key(fact: Fact) -> tuple[str, str]:
    return _canon(fact.subject), _canon(fact.predicate)


def build_fact_beliefs(facts: Iterable[Fact], *, as_of: float | None = None) -> list[FactBelief]:
    """Resolve raw facts into current beliefs plus historical alternatives.

    Facts stay immutable evidence. A belief is an interpretation of facts at a
    point in time, using validity windows and trust scores.
    """

    now = time.time() if as_of is None else float(as_of)
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)

    for fact in facts:
        if _fact_status(fact) == "retracted":
            continue
        if fact.subject and fact.predicate:
            groups[canonical_fact_key(fact)].append(fact)

    beliefs: list[FactBelief] = []

    for (subject, predicate), group in groups.items():
        scores = score_trust_recent_first(group)
        valid = [
            fact
            for fact in group
            if _is_valid_at(fact, now) and _fact_status(fact) not in {"historical", "retracted"}
        ]
        ranked_valid = sorted(valid, key=lambda fact: (scores.get(fact.id, 0.0), fact.provenance.time or 0.0), reverse=True)
        current = ranked_valid[0] if ranked_valid else None
        historical = [fact for fact in group if current is None or fact.id != current.id]
        alternatives = [
            fact
            for fact in historical
            if current is None or canonical_object(fact.object) != canonical_object(current.object)
        ]
        variants = sorted({fact.object for fact in group})

        if current is None:
            status = "historical_only"
            reason = "No valid fact for the requested time."
        elif alternatives:
            status = "conflict"
            reason = "Selected the highest-trust valid fact; alternatives remain as historical evidence."
        else:
            status = "current"
            reason = "Only one valid variant is available." if len(variants) == 1 else "Selected the highest-trust valid fact."

        beliefs.append(
            FactBelief(
                key=f"{subject}::{predicate}",
                subject=subject,
                predicate=predicate,
                current=current,
                alternatives=alternatives,
                historical=historical,
                variants=variants,
                current_score=scores.get(current.id, 0.0) if current else 0.0,
                status=status,
                reason=reason,
            )
        )

    beliefs.sort(key=lambda belief: belief.key)
    return beliefs


def canonical_object(value: str) -> str:
    return _canon(value)


def _canon(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def _float_meta(meta: dict[str, Any], key: str, *, default: float) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _is_valid_at(fact: Fact, as_of: float) -> bool:
    meta = fact.meta or {}

    valid_from = _optional_float(meta.get("valid_from"))
    valid_to = _optional_float(meta.get("valid_to"))
    expire_at = _optional_float(meta.get("expire_at"))

    if valid_from is not None and valid_from > as_of:
        return False

    if valid_to is not None and valid_to < as_of:
        return False

    if expire_at is not None and expire_at < as_of:
        return False

    return True


def _fact_status(fact: Fact) -> str:
    meta = fact.meta or {}
    if meta.get("superseded_by"):
        return "historical"
    return str(meta.get("status") or "current")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
