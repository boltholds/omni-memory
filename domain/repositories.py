from __future__ import annotations

from typing import Protocol

from domain.models import Fact


class IFactRepo(Protocol):
    def save_fact(self, fact: Fact) -> None:
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