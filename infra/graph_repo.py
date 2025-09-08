# infra/graph_repo.py
from __future__ import annotations

from typing import List, Any, Dict
import networkx as nx
import time

from domain.models import Fact, QuerySpec
from domain.ports import IGraphRepository


class GraphRepo(IGraphRepository):
    """
    Храним факты как рёбра MultiDiGraph:
    - узлы: subject и object (строки)
    - ребро: predicate (+ остальные поля факта)
    - ключ ребра = fact.id, чтобы избежать коллизий по одному s/p/o
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # ---- IGraphRepository ----
    def save_fact(self, fact: Fact) -> None:
        s = fact.subject
        o = fact.object
        p = fact.predicate

        # создаём/обновляем узлы
        if not self._g.has_node(s):
            self._g.add_node(s, type="entity")
        if not self._g.has_node(o):
            self._g.add_node(o, type="entity")

        # если ребро с таким id уже есть — обновим данные
        if self._g.has_edge(s, o, key=fact.id):
            self._g[s][o][fact.id].update(_fact_to_edge_attrs(fact))
        else:
            self._g.add_edge(s, o, key=fact.id, **_fact_to_edge_attrs(fact))

    def query(self, **query_spec: Any) -> List[Fact]:
        """
        Фильтр по равенству: subject / predicate / object.
        Любое поле можно опустить: вернём все, где остальные совпали.
        Пример: query(subject="Alice") или query(predicate="at")
        """
        spec: QuerySpec = {}
        # отфильтруем только известные ключи
        for k in ("subject", "predicate", "object"):
            v = query_spec.get(k)
            if v is not None:
                spec[k] = v  # type: ignore[assignment]

        results: List[Fact] = []
        # переберём все рёбра; MultiDiGraph хранит их как (u, v, key, data)
        for s, o, k, data in self._g.edges(keys=True, data=True):
            # быстрые отбрасывания
            if "predicate" not in data:
                continue
            if not _matches(s, data["predicate"], o, spec):
                continue
            results.append(_edge_to_fact(s, o, k, data))
        return results
    
    def gc_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        removed = 0
        to_remove = []
        for s, o, k, data in self._g.edges(keys=True, data=True):
            meta = data.get("meta") or {}
            exp = meta.get("expire_at")
            if exp is not None and float(exp) < now:
                to_remove.append((s, o, k))
        for s, o, k in to_remove:
            self._g.remove_edge(s, o, key=k)
            removed += 1
        return removed


# ----------------- helpers -----------------

def _fact_to_edge_attrs(f: Fact) -> Dict[str, Any]:
    return {
        "predicate": f.predicate,
        "provenance": f.provenance.model_dump(),
        "meta": f.meta,
    }

def _edge_to_fact(s: str, o: str, k: str, data: Dict[str, Any]) -> Fact:
    return Fact(
        id=k,
        subject=s,
        predicate=str(data.get("predicate", "")),
        object=o,
        # допускаем, что в графе лежит dict -> в модель
        provenance=data.get("provenance"),  # pydantic сам приведёт
        meta=data.get("meta", {}) or {},
    )

def _matches(s: str, p: str, o: str, spec: QuerySpec) -> bool:
    sj = spec.get("subject")
    pj = spec.get("predicate")
    oj = spec.get("object")
    if sj is not None and s != sj:
        return False
    if pj is not None and p != pj:
        return False
    if oj is not None and o != oj:
        return False
    return True
