from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


GraphEdge = tuple[str, str, str, dict[str, Any]]


@runtime_checkable
class GraphBackend(Protocol):
    """Small facade over concrete graph storage libraries.

    Graph repositories should depend on this protocol instead of depending on
    NetworkX or another graph database client directly.
    """

    def edge_count(self) -> int: ...

    def clear(self) -> int: ...

    def upsert_edge(self, subject: str, object_: str, key: str, attrs: dict[str, Any]) -> None: ...

    def get_edge(self, subject: str, object_: str, key: str) -> dict[str, Any] | None: ...

    def remove_edge(self, subject: str, object_: str, key: str) -> bool: ...

    def iter_edges(self) -> Iterable[GraphEdge]: ...


def build_graph_backend(prototype: GraphBackend | None = None) -> GraphBackend:
    return prototype or NetworkxGraphBackend()


class NetworkxGraphBackend:
    def __init__(self) -> None:
        import networkx as nx

        self._g = nx.MultiDiGraph()

    def edge_count(self) -> int:
        return int(self._g.number_of_edges())

    def clear(self) -> int:
        removed = self.edge_count()
        self._g.clear()
        return removed

    def upsert_edge(self, subject: str, object_: str, key: str, attrs: dict[str, Any]) -> None:
        if not self._g.has_node(subject):
            self._g.add_node(subject, type="entity")
        if not self._g.has_node(object_):
            self._g.add_node(object_, type="entity")

        if self._g.has_edge(subject, object_, key=key):
            self._g[subject][object_][key].clear()
            self._g[subject][object_][key].update(attrs)
        else:
            self._g.add_edge(subject, object_, key=key, **attrs)

    def get_edge(self, subject: str, object_: str, key: str) -> dict[str, Any] | None:
        if not self._g.has_edge(subject, object_, key=key):
            return None
        return dict(self._g[subject][object_][key])

    def remove_edge(self, subject: str, object_: str, key: str) -> bool:
        if not self._g.has_edge(subject, object_, key=key):
            return False
        self._g.remove_edge(subject, object_, key=key)
        return True

    def iter_edges(self) -> Iterable[GraphEdge]:
        for subject, object_, key, data in self._g.edges(keys=True, data=True):
            yield str(subject), str(object_), str(key), dict(data)
