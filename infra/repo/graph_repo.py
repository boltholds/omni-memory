# infra/graph_repo.py
from __future__ import annotations

from typing import List, Any, Dict
import networkx as nx
import time

from domain.models import Fact, QuerySpec
from domain.ports import IGraphRepository
from app.profiling import timed
from app.metrics import GRAPH_FACTS


class GraphRepo(IGraphRepository):
    """
    Храним факты как рёбра MultiDiGraph:
    - узлы: subject и object (строки)
    - ребро: predicate (+ остальные поля факта)
    - ключ ребра = fact.id, чтобы избежать коллизий по одному s/p/o
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    
    def count(self) -> int:
        return self._g.number_of_edges()

    def clear(self) -> int:
        removed = self.count()
        self._g.clear()
        try:
            GRAPH_FACTS.set(self.count())
        except Exception:
            pass
        return removed
    
    # ---- IGraphRepository ----
    def save_fact(self, fact: Fact) -> None:
        s = fact.subject
        o = fact.object
        p = fact.predicate

        existing = self._find_edge_by_id(fact.id)
        if existing is not None and existing[:2] != (s, o):
            old_s, old_o, old_key = existing
            self._g.remove_edge(old_s, old_o, key=old_key)

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
        try:
            GRAPH_FACTS.set(self.count())
        except Exception:
            pass

    def get_fact(self, fact_id: str) -> Fact | None:
        edge = self._find_edge_by_id(fact_id)
        if edge is None:
            return None
        s, o, key = edge
        return _edge_to_fact(s, o, key, self._g[s][o][key])

    def remove_fact(self, fact_id: str) -> bool:
        edge = self._find_edge_by_id(fact_id)
        if edge is None:
            return False
        s, o, key = edge
        self._g.remove_edge(s, o, key=key)
        try:
            GRAPH_FACTS.set(self.count())
        except Exception:
            pass
        return True

    @timed("retriever.retrieve", slow_ms=100)
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
        try:
            GRAPH_FACTS.set(self.count())
        except Exception:
            pass
        finally:
            return removed

    def _find_edge_by_id(self, fact_id: str) -> tuple[str, str, str] | None:
        for s, o, key in self._g.edges(keys=True):
            if key == fact_id:
                return str(s), str(o), str(key)
        return None
        

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
