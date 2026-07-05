from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


RecordT = TypeVar("RecordT", bound=BaseModel)


@runtime_checkable
class RecordStoreBackend(Protocol[RecordT]):
    """Storage facade for id-addressable typed records."""

    def save(self, record_id: str, record: RecordT) -> None: ...
    def get(self, record_id: str) -> RecordT | None: ...
    def values(self) -> list[RecordT]: ...
    def count(self) -> int: ...
    def clear(self) -> int: ...


@dataclass(frozen=True)
class RecordStoreBackends:
    decision: RecordStoreBackend[Any] | None = None
    experience: RecordStoreBackend[Any] | None = None
    skill: RecordStoreBackend[Any] | None = None
    failure_pattern: RecordStoreBackend[Any] | None = None
    review_queue: RecordStoreBackend[Any] | None = None


class InMemoryRecordStoreBackend(Generic[RecordT]):
    def __init__(self, initial: Iterable[RecordT] | None = None) -> None:
        self._store: dict[str, RecordT] = {}
        for record in initial or []:
            self.save(str(record.id), record)

    def save(self, record_id: str, record: RecordT) -> None:
        self._store[str(record_id)] = record

    def get(self, record_id: str) -> RecordT | None:
        return self._store.get(str(record_id))

    def values(self) -> list[RecordT]:
        return list(self._store.values())

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> int:
        removed = len(self._store)
        self._store.clear()
        return removed


class JsonRecordStoreBackend(Generic[RecordT]):
    def __init__(self, path: str | Path, model_type: type[RecordT]) -> None:
        self.path = Path(path)
        self.model_type = model_type
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._inner = InMemoryRecordStoreBackend[RecordT]()
        self._load()

    def save(self, record_id: str, record: RecordT) -> None:
        self._inner.save(record_id, record)
        self._flush()

    def get(self, record_id: str) -> RecordT | None:
        return self._inner.get(record_id)

    def values(self) -> list[RecordT]:
        return self._inner.values()

    def count(self) -> int:
        return self._inner.count()

    def clear(self) -> int:
        removed = self._inner.clear()
        self._flush()
        return removed

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        for item in raw:
            record = self.model_type.model_validate(item)
            self._inner.save(str(record.id), record)

    def _flush(self) -> None:
        data = [record.model_dump(mode="json") for record in self._inner.values()]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
