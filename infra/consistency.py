# infra/consistency.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from domain.models import ConflictItem, ConflictReport, Fact
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
        scores[f.id] = 0.5 * t_norm + 0.5 * s_norm
    return scores
