from __future__ import annotations

from domain.models import DomainLink, DomainNode


class DomainGraphRepo:
    def __init__(self) -> None:
        self._nodes: dict[str, DomainNode] = {}
        self._links: list[DomainLink] = []

    def upsert_node(self, node: DomainNode) -> DomainNode:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            return node
        merged = existing.model_copy(
            update={
                "name": node.name or existing.name,
                "kind": node.kind or existing.kind,
                "aliases": _unique([*existing.aliases, *node.aliases]),
                "meta": {**existing.meta, **node.meta},
            }
        )
        self._nodes[node.id] = merged
        return merged

    def get_node(self, node_id: str) -> DomainNode | None:
        return self._nodes.get(node_id)

    def list_nodes(self, *, kind: str | None = None) -> list[DomainNode]:
        items = list(self._nodes.values())
        if kind is not None:
            items = [item for item in items if item.kind == kind]
        return sorted(items, key=lambda item: (item.kind, item.id))

    def add_link(self, link: DomainLink) -> DomainLink:
        for existing in self._links:
            if (
                existing.source_id == link.source_id
                and existing.relation == link.relation
                and existing.target_id == link.target_id
            ):
                return existing
        self._links.append(link)
        return link

    def list_links(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relation: str | None = None,
    ) -> list[DomainLink]:
        items = list(self._links)
        if source_id is not None:
            items = [item for item in items if item.source_id == source_id]
        if target_id is not None:
            items = [item for item in items if item.target_id == target_id]
        if relation is not None:
            items = [item for item in items if item.relation == relation]
        return sorted(items, key=lambda item: (item.source_id, item.relation, item.target_id))

    def related_domain_ids(self, domain_id: str) -> list[str]:
        related: list[str] = []
        for link in self._links:
            if link.source_id == domain_id:
                related.append(link.target_id)
            elif link.target_id == domain_id:
                related.append(link.source_id)
        return _unique(related)

    def count(self) -> int:
        return len(self._nodes)

    def link_count(self) -> int:
        return len(self._links)

    def clear(self) -> int:
        removed = len(self._nodes) + len(self._links)
        self._nodes.clear()
        self._links.clear()
        return removed


def domain_id(kind: str, name: str) -> str:
    slug = str(name or "").strip().lower().replace(" ", "-").replace("_", "-")
    slug = "-".join(part for part in slug.split("-") if part)
    return f"domain:{kind}:{slug or 'unnamed'}"


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out
