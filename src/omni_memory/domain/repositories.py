from __future__ import annotations

from typing import Protocol

from omni_memory.domain.models import Fact, MemoryObject


class IFactRepo(Protocol):
    def save_fact(self, fact: Fact) -> None:
        ...

    def get_fact(self, fact_id: str) -> Fact | None:
        ...

    def remove_fact(self, fact_id: str) -> bool:
        ...

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
    ) -> list[Fact]:
        ...

    def count(self) -> int:
        ...

    def find_conflicts(self):
        ...
        
        
 


class IVectorRepo(Protocol):
    def save_object(self, obj: MemoryObject) -> bool:
        ...

    def semantic_search(self, text: str, k: int = 5) -> list[MemoryObject]:
        ...

    def is_duplicate_text(self, text: str) -> bool:
        ...

    def count(self) -> int:
        ...
