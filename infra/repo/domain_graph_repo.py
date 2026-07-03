from __future__ import annotations

from typing import Any

import networkx as nx

from domain.models import DomainLink, DomainNode


class DomainGraphRepo:
    """In-memory domain graph backed by NetworkX.

    The graph is intentionally close to `GraphRepo`: nodes are domain ids and
    links are directed MultiDiGraph edges. This keeps the MVP small while making
    future domain-aware traversal/ranking straightforward.
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    def upsert_node(self, node: DomainNode) -> DomainNode:
        existing = self.get_node(node.id)
        if existing is None:
            self._g.add_node(node.id, **_node_to_attrs(node))
            return node

        merged = existing.model_copy(
            update={
                "name": node.name or existing.name,
                "kind": node.kind or existing.kind,
                "aliases": _unique([*existing.aliases, *node.aliases]),
                "meta": {**existing.meta, **node.meta},
            }
        )
        self._g.add_node(node.id, **_node_to_attrs(merged))
        return merged

    def get_node(self, node_id: str) -> DomainNode | None:
        if not self._g.has_node(node_id):
            return None
        return _attrs_to_node(str(node_id), self._g.nodes[node_id])

    def list_nodes(self, *, kind: str | None = None) -> list[DomainNode]:
        items = [_attrs_to_node(str(node_id), data) for node_id, data in self._g.nodes(data=True)]
        if kind is not None:
            items = [item for item in items if item.kind == kind]
        return sorted(items, key=lambda item: (item.kind, item.id))

    def add_link(self, link: DomainLink) -> DomainLink:
        self._ensure_node(link.source_id)
        self._ensure_node(link.target_id)
        key = _edge_key(link)
        if self._g.has_edge(link.source_id, link.target_id, key=key):
            return _edge_to_link(
                link.source_id,
                link.target_id,
                self._g[link.source_id][link.target_id][key],
            )
        self._g.add_edge(link.source_id, link.target_id, key=key, **_link_to_edge_attrs(link))
        return link

    def list_links(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relation: str | None = None,
    ) -> list[DomainLink]:
        items: list[DomainLink] = []
        for source, target, _key, data in self._g.edges(keys=True, data=True):
            link = _edge_to_link(str(source), str(target), data)
            if source_id is not None and link.source_id != source_id:
                continue
            if target_id is not None and link.target_id != target_id:
                continue
            if relation is not None and link.relation != relation:
                continue
            items.append(link)
        return sorted(items, key=lambda item: (item.source_id, item.relation, item.target_id))

    def related_domain_ids(self, domain_id: str) -> list[str]:
        if not self._g.has_node(domain_id):
            return []
        related = [str(node_id) for node_id in self._g.successors(domain_id)]
        related.extend(str(node_id) for node_id in self._g.predecessors(domain_id))
        return _unique(related)

    def successors(self, domain_id: str, *, relation: str | None = None) -> list[str]:
        if not self._g.has_node(domain_id):
            return []
        out: list[str] = []
        for target_id in self._g.successors(domain_id):
            edge_data = self._g.get_edge_data(domain_id, target_id) or {}
            if relation is None or any(data.get("relation") == relation for data in edge_data.values()):
                out.append(str(target_id))
        return sorted(_unique(out))

    def predecessors(self, domain_id: str, *, relation: str | None = None) -> list[str]:
        if not self._g.has_node(domain_id):
            return []
        out: list[str] = []
        for source_id in self._g.predecessors(domain_id):
            edge_data = self._g.get_edge_data(source_id, domain_id) or {}
            if relation is None or any(data.get("relation") == relation for data in edge_data.values()):
                out.append(str(source_id))
        return sorted(_unique(out))

    def reachable_domain_ids(self, domain_id: str, *, max_depth: int = 2) -> list[str]:
        """Return nearby domain ids in BFS order, closest domains first."""
        if not self._g.has_node(domain_id) or max_depth < 1:
            return []
        visited: set[str] = {domain_id}
        frontier: list[str] = [domain_id]
        ordered: list[str] = []
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for item in frontier:
                neighbours = [str(node_id) for node_id in self._g.successors(item)]
                neighbours.extend(str(node_id) for node_id in self._g.predecessors(item))
                for neighbour in _unique(neighbours):
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    ordered.append(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return ordered

    def count(self) -> int:
        return self._g.number_of_nodes()

    def link_count(self) -> int:
        return self._g.number_of_edges()

    def clear(self) -> int:
        removed = self.count() + self.link_count()
        self._g.clear()
        return removed

    def _ensure_node(self, node_id: str) -> None:
        if not self._g.has_node(node_id):
            self._g.add_node(node_id, **_node_to_attrs(DomainNode(id=node_id, name=node_id)))


def domain_id(kind: str, name: str) -> str:
    slug = str(name or "").strip().lower().replace(" ", "-").replace("_", "-")
    slug = "-".join(part for part in slug.split("-") if part)
    return f"domain:{kind}:{slug or 'unnamed'}"


def _node_to_attrs(node: DomainNode) -> dict[str, Any]:
    return node.model_dump(mode="json")


def _attrs_to_node(node_id: str, data: dict[str, Any]) -> DomainNode:
    payload = dict(data or {})
    payload.setdefault("id", node_id)
    payload.setdefault("name", node_id)
    payload.setdefault("kind", "knowledge_area")
    payload.setdefault("aliases", [])
    payload.setdefault("meta", {})
    return DomainNode.model_validate(payload)


def _link_to_edge_attrs(link: DomainLink) -> dict[str, Any]:
    return link.model_dump(mode="json")


def _edge_to_link(source_id: str, target_id: str, data: dict[str, Any]) -> DomainLink:
    payload = dict(data or {})
    payload.setdefault("source_id", source_id)
    payload.setdefault("target_id", target_id)
    payload.setdefault("relation", "related_to")
    payload.setdefault("confidence", 1.0)
    payload.setdefault("meta", {})
    return DomainLink.model_validate(payload)


def _edge_key(link: DomainLink) -> str:
    return link.relation


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out
