# app/retriever.py
from __future__ import annotations

from typing import List, Set

from domain.models import RetrievalBundle, MemoryObject, Fact, Episode
from domain.ports import IRetriever, IMemoryReadRepository, IGraphRepository, IEpisodicRepository




def _simple_entities(query: str) -> List[str]:
    """
    Примитивная NER-заглушка:
    - токены из букв/цифр длиной >= 3
    - без дубликатов, регистр игнорируем
    """
    seen: Set[str] = set()
    ents: List[str] = []
    for raw in query.replace("_", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum()).lower()
        if len(tok) >= 3 and tok not in seen:
            seen.add(tok)
            ents.append(tok)
    return ents


class Retriever(IRetriever):
    """
    Объединённый извлекатель:
      - semantic: векторный поиск по исходному запросу
      - graph: выборка фактов по найденным сущностям (subject/object)
      - episodic: поиск эпизодов по сущностям
    """

    def __init__(
        self,
        vector_repo: IMemoryReadRepository,
        graph_repo: IGraphRepository,
        episodic_repo: IEpisodicRepository,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo

    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        ents = _simple_entities(query)

        # I Семантические чанки
        semantic_chunks: List[MemoryObject] = self._vector.semantic_search(query, k=k_sem)

        # II Факты (по subject и по object для каждой выделенной сущности)
        facts: List[Fact] = []
        seen_ids: Set[str] = set()
        for e in ents:
            for f in self._graph.query(subject=e):
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    facts.append(f)
            for f in self._graph.query(object=e):
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    facts.append(f)

        # III Эпизоды (пользователя пока не извлекаем -> None)
        episodes: List[Episode] = self._episodic.search(user=None, entities=ents, k=k_eps)

        return RetrievalBundle(
            semantic_chunks=semantic_chunks,
            facts=facts,
            episodes=episodes,
            citations=[],
        )
