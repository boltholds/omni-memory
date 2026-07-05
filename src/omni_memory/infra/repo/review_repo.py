from __future__ import annotations

import json
from pathlib import Path

from omni_memory.domain.models import ReviewItem


class ReviewQueueRepo:
    def __init__(self) -> None:
        self._store: dict[str, ReviewItem] = {}

    def save_review_item(self, item: ReviewItem) -> None:
        self._store[item.id] = item

    def get_review_item(self, item_id: str) -> ReviewItem | None:
        return self._store.get(item_id)

    def list_review_items(
        self,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewItem]:
        items = list(self._store.values())
        if status:
            items = [item for item in items if item.status == status]
        if kind:
            items = [item for item in items if item.kind == kind]
        items.sort(key=lambda item: item.provenance.time or 0.0, reverse=True)
        return items[: max(0, limit)] if limit is not None else items

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = len(self._store)
        self._store.clear()
        return removed


class PersistentReviewQueueRepo:
    def __init__(self, inner: ReviewQueueRepo, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def save_review_item(self, item: ReviewItem) -> None:
        self.inner.save_review_item(item)
        self._flush()

    def get_review_item(self, item_id: str) -> ReviewItem | None:
        return self.inner.get_review_item(item_id)

    def list_review_items(
        self,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewItem]:
        return self.inner.list_review_items(status=status, kind=kind, limit=limit)

    def count(self) -> int:
        return self.inner.count()

    def clear(self) -> int:
        removed = self.inner.clear()
        self._flush()
        return removed

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        for item in raw:
            self.inner.save_review_item(ReviewItem.model_validate(item))

    def _flush(self) -> None:
        data = [item.model_dump(mode="json") for item in self.inner.list_review_items()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
