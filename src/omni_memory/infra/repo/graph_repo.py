# infra/graph_repo.py
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from omni_memory.metrics import GRAPH_FACTS
from omni_memory.profiling import timed
from omni_memory.domain.models import Fact, QuerySpec
from omni_memory.domain.ports import IGraphRepository
from omni_memory.infra.graph_backend import GraphBackend, build_graph_backend


class GraphRepo(IGraphRepository):
    """
    Semantic fact repository over an injectable graph backend.

    Facts are stored as graph edges:
    - nodes: subject and object strings
    - edge key: fact.id
    - edge attrs: predicate, provenance, meta

    Query path is indexed by subject/object/predicate/id inside the repository.
    The concrete graph library is hidden behind GraphBackend.
    """

    def __init__(self, backend: GraphBackend | None = None) -> None:
        self._backend = build_graph_backend(backend)
        self._edge_by_id: dict[str, tuple[str, str, str]] = {}
        self._by_subject: dict[str, set[str]] = {}
        self._by_object: dict[str, set[str]] = {}
        self._by_predicate: dict[str, set[str]] = {}

    def count(self) -> int:
        return self._backend.edge_count()

    def clear(self) -> int:
        removed = self._backend.clear()
        self._edge_by_id.clear()
        self._by_subject.clear()
        self._by_object.clear()
        self._by_predicate.clear()
        self._set_metric()
        return removed

    def save_fact(self, fact: Fact) -> None:
        s = fact.subject
        o = fact.object

        existing = self._edge_by_id.get(fact.id)
        if existing is not None:
            old_s, old_o, old_key = existing
            old_data = self._backend.get_edge(old_s, old_o, old_key) or {}
            self._unindex_edge(old_s, old_o, old_key, old_data)
            if (old_s, old_o) != (s, o):
                self._backend.remove_edge(old_s, old_o, old_key)

        attrs = _fact_to_edge_attrs(fact)
        self._backend.upsert_edge(s, o, fact.id, attrs)
        self._index_edge(s, o, fact.id, attrs)
        self._set_metric()

    def get_fact(self, fact_id: str) -> Fact | None:
        edge = self._edge_by_id.get(fact_id)
        if edge is None:
            return None
        s, o, key = edge
        data = self._backend.get_edge(s, o, key)
        if data is None:
            return None
        return _edge_to_fact(s, o, key, data)

    def remove_fact(self, fact_id: str) -> bool:
        edge = self._edge_by_id.get(fact_id)
        if edge is None:
            return False
        s, o, key = edge
        data = self._backend.get_edge(s, o, key)
        if data is None:
            self._edge_by_id.pop(fact_id, None)
            return False
        self._unindex_edge(s, o, key, data)
        removed = self._backend.remove_edge(s, o, key)
        self._set_metric()
        return removed

    @timed("retriever.retrieve", slow_ms=100)
    def query(self, **query_spec: Any) -> List[Fact]:
        """
        Filter by equality: subject / predicate / object.
        Any field may be omitted. Indexed queries are used when at least one
        indexed field is provided.
        """
        spec: QuerySpec = {}
        for key in ("subject", "predicate", "object"):
            value = query_spec.get(key)
            if value is not None:
                spec[key] = value  # type: ignore[assignment]

        edge_ids = self._candidate_edge_ids(spec)
        edge_iter: Iterable[tuple[str, str, str, dict[str, Any]]]
        if edge_ids is None:
            edge_iter = self._backend.iter_edges()
        else:
            edge_iter = self._edges_by_ids(edge_ids)

        results: List[Fact] = []
        for s, o, key, data in edge_iter:
            if "predicate" not in data:
                continue
            if not _matches(str(s), str(data["predicate"]), str(o), spec):
                continue
            results.append(_edge_to_fact(str(s), str(o), str(key), data))
        return results

    def query_entity_neighborhood(self, entities: list[str], *, include_incoming: bool = True, include_outgoing: bool = True) -> List[Fact]:
        """Return all facts adjacent to any entity using subject/object indexes."""
        ids: set[str] = set()
        for entity in entities:
            if include_outgoing:
                ids.update(self._by_subject.get(entity, set()))
            if include_incoming:
                ids.update(self._by_object.get(entity, set()))
        return [_edge_to_fact(s, o, key, data) for s, o, key, data in self._edges_by_ids(ids)]

    def gc_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        to_remove: list[str] = []
        for _s, _o, key, data in self._backend.iter_edges():
            meta = data.get("meta") or {}
            exp = meta.get("expire_at")
            if exp is not None and float(exp) < now:
                to_remove.append(str(key))
        removed = 0
        for fact_id in to_remove:
            if self.remove_fact(fact_id):
                removed += 1
        self._set_metric()
        return removed

    def _candidate_edge_ids(self, spec: QuerySpec) -> set[str] | None:
        sets: list[set[str]] = []
        subject = spec.get("subject")
        predicate = spec.get("predicate")
        object_ = spec.get("object")
        if subject is not None:
            sets.append(set(self._by_subject.get(str(subject), set())))
        if predicate is not None:
            sets.append(set(self._by_predicate.get(str(predicate), set())))
        if object_ is not None:
            sets.append(set(self._by_object.get(str(object_), set())))
        if not sets:
            return None
        result = sets[0]
        for item in sets[1:]:
            result &= item
        return result

    def _edges_by_ids(self, edge_ids: Iterable[str]) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        for edge_id in edge_ids:
            edge = self._edge_by_id.get(edge_id)
            if edge is None:
                continue
            s, o, key = edge
            data = self._backend.get_edge(s, o, key)
            if data is not None:
                yield s, o, key, data

    def _index_edge(self, s: str, o: str, key: str, data: dict[str, Any]) -> None:
        predicate = str(data.get("predicate", ""))
        self._edge_by_id[key] = (s, o, key)
        self._by_subject.setdefault(s, set()).add(key)
        self._by_object.setdefault(o, set()).add(key)
        self._by_predicate.setdefault(predicate, set()).add(key)

    def _unindex_edge(self, s: str, o: str, key: str, data: dict[str, Any]) -> None:
        predicate = str(data.get("predicate", ""))
        self._edge_by_id.pop(key, None)
        _discard_index(self._by_subject, s, key)
        _discard_index(self._by_object, o, key)
        _discard_index(self._by_predicate, predicate, key)

    def _find_edge_by_id(self, fact_id: str) -> tuple[str, str, str] | None:
        return self._edge_by_id.get(fact_id)

    def _set_metric(self) -> None:
        try:
            GRAPH_FACTS.set(self.count())
        except Exception:
            pass


# ----------------- helpers -----------------

def _discard_index(index: dict[str, set[str]], value: str, key: str) -> None:
    bucket = index.get(value)
    if bucket is None:
        return
    bucket.discard(key)
    if not bucket:
        index.pop(value, None)


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
        provenance=data.get("provenance"),
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
