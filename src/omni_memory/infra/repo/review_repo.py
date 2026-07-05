from __future__ import annotations

from pathlib import Path

from omni_memory.domain.models import ReviewItem
from omni_memory.infra.record_store import InMemoryRecordStoreBackend, JsonRecordStoreBackend, RecordStoreBackend


class ReviewQueueRepo:
    def __init__(self, backend: RecordStoreBackend[ReviewItem] | None = None) -> None:
        self._backend = backend or InMemoryRecordStoreBackend[ReviewItem]()

    def save_review_item(self, item: ReviewItem) -> None:
        self._backend.save(item.id, item)

    def get_review_item(self, item_id: str) -> ReviewItem | None:
        return self._backend.get(item_id)

    def list_review_items(
        self,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewItem]:
        items = self._backend.values()
        if status:
            items = [item for item in items if item.status == status]
        if kind:
            items = [item for item in items if item.kind == kind]
        items.sort(key=lambda item: item.provenance.time or 0.0, reverse=True)
        return items[: max(0, limit)] if limit is not None else items

    def count(self) -> int:
        return self._backend.count()

    def clear(self) -> int:
        return self._backend.clear()


class PersistentReviewQueueRepo(ReviewQueueRepo):
    def __init__(self, inner: ReviewQueueRepo, path: str | Path) -> None:
        super().__init__(backend=JsonRecordStoreBackend(path, ReviewItem))
        for item in inner.list_review_items():
            self.save_review_item(item)
