from __future__ import annotations

from typing import Any, Protocol

from omni_memory.domain.models import ContextPack, RetrievalBundle, WriteReport
from omni_memory.domain.requests import RecordExperienceRequest


class MemoryAnswerLike(Protocol):
    answer: str
    advisories: list[str]
    used_sections: list[str]
    context: dict[str, Any]
    model: str | None


class LangChainMemoryReader(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        k_sem: int = 5,
        k_eps: int = 3,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        ...

    def build_context(
        self,
        query: str,
        *,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> ContextPack:
        ...

    def ask(
        self,
        question: str,
        *,
        lang: str = "en",
        style: str = "concise",
        temperature: float | None = None,
        include_context: bool = True,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> MemoryAnswerLike:
        ...


class LangChainMemoryWriter(Protocol):
    def write_items(
        self,
        items: list[dict[str, Any]],
        *,
        source: str = "langchain",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> WriteReport:
        ...

    def write_note(
        self,
        text: str,
        *,
        source: str = "langchain",
        meta: dict[str, Any] | None = None,
    ) -> WriteReport:
        ...

    def write_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        source: str = "langchain",
        confidence: float = 1.0,
    ) -> WriteReport:
        ...

    def record_experience(
        self,
        request: RecordExperienceRequest,
    ) -> WriteReport:
        ...


class LangChainMemory(LangChainMemoryReader, LangChainMemoryWriter, Protocol):
    pass
